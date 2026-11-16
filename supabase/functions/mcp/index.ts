// Supabase Edge Function: Free-Tier Hybrid RAG Engine (Raggae)
// Runtime: Deno / Supabase Edge Runtime
// Protocol: Model Context Protocol (MCP) over Server-Sent Events (SSE) & Streamable HTTP
// Features: Hybrid Search (pgvector HNSW + FTS GIN) + Batch Embedding + Free-Tier Rate Protection

import { createClient, SupabaseClient } from "npm:@supabase/supabase-js@^2.49.1";

interface Session {
  send: (data: any) => void;
  close: () => void;
}

// Active SSE sessions in memory
const sessions = new Map<string, Session>();

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Tool Definitions according to MCP specification
const MCP_TOOLS = [
  {
    name: "rag_search",
    description: "Search indexed knowledge using hybrid vector similarity and keyword full-text search with metadata filters",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Search query or natural language question" },
        collection: { type: "string", description: "Optional collection/namespace filter (e.g. 'docs', 'code', 'research')" },
        limit: { type: "number", description: "Max number of chunks to return", default: 8 },
        match_threshold: { type: "number", description: "Minimum cosine similarity threshold (0.0 to 1.0)", default: 0.2 },
        hybrid: { type: "boolean", description: "Whether to run hybrid retrieval (Vector + Full-Text) or vector only", default: true },
      },
      required: ["query"],
    },
  },
  {
    name: "rag_ingest_document",
    description: "Ingest a document or text content into the RAG store with automatic chunking, deduplication, and embedding",
    inputSchema: {
      type: "object",
      properties: {
        title: { type: "string", description: "Document title or file identifier" },
        content: { type: "string", description: "Full document text (Markdown, source code, text)" },
        collection: { type: "string", description: "Target collection/namespace", default: "default" },
        source_uri: { type: "string", description: "Source URL or file path" },
        chunk_size: { type: "number", description: "Max characters per chunk", default: 1000 },
        chunk_overlap: { type: "number", description: "Character overlap between consecutive chunks", default: 150 },
        metadata: { type: "object", description: "Arbitrary JSON metadata for document and chunks" },
      },
      required: ["title", "content"],
    },
  },
  {
    name: "rag_list_documents",
    description: "List indexed documents, chunk counts, collections, and storage timestamps",
    inputSchema: {
      type: "object",
      properties: {
        collection: { type: "string", description: "Optional collection filter" },
        limit: { type: "number", description: "Max documents to return", default: 25 },
      },
    },
  },
  {
    name: "rag_get_document",
    description: "Retrieve full document metadata and its indexed chunks in sequential order",
    inputSchema: {
      type: "object",
      properties: {
        document_id: { type: "string", description: "Document UUID" },
        title: { type: "string", description: "Document title to look up if UUID is unknown" },
        include_chunks: { type: "boolean", description: "Whether to include chunks in response", default: true },
      },
    },
  },
  {
    name: "rag_delete_document",
    description: "Delete an indexed document and its associated chunks from the database",
    inputSchema: {
      type: "object",
      properties: {
        document_id: { type: "string", description: "Document UUID to delete" },
        source_uri: { type: "string", description: "Source URI to delete all matching documents" },
      },
    },
  },
  {
    name: "rag_storage_stats",
    description: "Check database storage metrics, chunk counts, and Supabase 500MB free tier capacity",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
];

/**
 * SHA-256 Hash helper for content deduplication & quota saving
 */
async function sha256Hex(text: string): Promise<string> {
  const clean = text.replace(/\s+/g, " ").trim();
  const msgBuffer = new TextEncoder().encode(clean);
  const hashBuffer = await crypto.subtle.digest("SHA-256", msgBuffer);
  return Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Recursive character text splitter for intelligent chunking
 */
function splitTextIntoChunks(
  text: string,
  chunkSize: number = 1000,
  chunkOverlap: number = 150
): string[] {
  if (!text || text.trim().length === 0) return [];
  if (text.length <= chunkSize) return [text.trim()];

  const separators = ["\n\n", "\n", "```", ". ", "; ", " ", ""];
  
  function split(textToSplit: string, sepIndex: number): string[] {
    if (textToSplit.length <= chunkSize || sepIndex >= separators.length) {
      return [textToSplit.trim()].filter(Boolean);
    }

    const sep = separators[sepIndex];
    const rawParts = sep ? textToSplit.split(sep) : Array.from(textToSplit);
    const result: string[] = [];
    let current = "";

    for (const part of rawParts) {
      const candidate = current ? current + sep + part : part;
      if (candidate.length <= chunkSize) {
        current = candidate;
      } else {
        if (current) result.push(current.trim());
        if (part.length > chunkSize) {
          // Recurse with finer separator
          result.push(...split(part, sepIndex + 1));
          current = "";
        } else {
          current = part;
        }
      }
    }

    if (current) result.push(current.trim());
    return result.filter(Boolean);
  }

  const rawChunks = split(text, 0);

  // Apply overlap if multiple chunks
  if (rawChunks.length <= 1 || chunkOverlap <= 0) {
    return rawChunks;
  }

  const overlapped: string[] = [];
  for (let i = 0; i < rawChunks.length; i++) {
    if (i === 0) {
      overlapped.push(rawChunks[i]);
    } else {
      const prev = rawChunks[i - 1];
      const overlapText = prev.slice(-chunkOverlap);
      overlapped.push(`${overlapText}\n${rawChunks[i]}`.trim());
    }
  }

  return overlapped;
}

/**
 * Batch generate vector embeddings using Gemini text-embedding-004 (or fallback to OpenAI)
 */
async function generateEmbeddingsBatch(
  texts: string[]
): Promise<Array<{ embedding: number[] | null; status: string }>> {
  if (texts.length === 0) return [];

  const geminiKey = Deno.env.get("GEMINI_API_KEY") || Deno.env.get("GOOGLE_API_KEY");
  const openaiKey = Deno.env.get("OPENAI_API_KEY");

  // 1. Google Gemini text-embedding-004 (batchEmbedContents endpoint)
  if (geminiKey) {
    try {
      const requests = texts.map((t) => ({
        model: "models/text-embedding-004",
        content: { parts: [{ text: t.replace(/\n/g, " ").trim() || " " }] },
        outputDimensionality: 768,
      }));

      const res = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key=${geminiKey}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ requests }),
        }
      );

      if (res.ok) {
        const data: any = await res.json();
        const embeddings = data.embeddings || [];
        return texts.map((_, idx) => {
          const emb = embeddings[idx]?.values || null;
          return {
            embedding: emb,
            status: emb ? `gemini_ok (${emb.length} dims)` : "gemini_empty",
          };
        });
      } else {
        const errText = await res.text();
        console.error("Gemini batch embedding error:", res.status, errText);
        if (res.status === 429) {
          return texts.map(() => ({ embedding: null, status: "gemini_429_quota_exceeded" }));
        }
      }
    } catch (err: any) {
      console.error("Gemini batch embedding failed:", err);
    }
  }

  // 2. Fallback: OpenAI text-embedding-3-small (dimensions=768)
  if (openaiKey) {
    try {
      const cleanInputs = texts.map((t) => t.replace(/\n/g, " ").trim() || " ");
      const res = await fetch("https://api.openai.com/v1/embeddings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${openaiKey}`,
        },
        body: JSON.stringify({
          input: cleanInputs,
          model: "text-embedding-3-small",
          dimensions: 768,
        }),
      });

      if (res.ok) {
        const data: any = await res.json();
        const items = data.data || [];
        return texts.map((_, idx) => ({
          embedding: items[idx]?.embedding || null,
          status: items[idx]?.embedding ? "openai_ok (768 dims)" : "openai_empty",
        }));
      }
    } catch (err: any) {
      console.error("OpenAI embedding fallback failed:", err);
    }
  }

  return texts.map(() => ({ embedding: null, status: "no_provider_available" }));
}

/**
 * Single text embedding generator for search queries
 */
async function generateSingleEmbedding(text: string): Promise<number[] | null> {
  const results = await generateEmbeddingsBatch([text]);
  return results[0]?.embedding || null;
}

/**
 * Execute MCP tool operations
 */
async function executeTool(name: string, args: any, supabase: SupabaseClient): Promise<any> {
  try {
    switch (name) {
      case "rag_search": {
        const {
          query,
          collection = null,
          limit = 8,
          match_threshold = 0.2,
          hybrid = true,
        } = args;

        if (!query || typeof query !== "string") {
          return { isError: true, content: [{ type: "text", text: "Argument 'query' is required." }] };
        }

        // Generate query embedding
        const queryEmbedding = await generateSingleEmbedding(query);

        if (hybrid) {
          // Use hybrid search RPC (vector + FTS via RRF)
          const { data, error } = await supabase.rpc("match_document_chunks_hybrid", {
            query_text: query,
            query_embedding: queryEmbedding,
            match_count: limit,
            filter_collection: collection || null,
          });

          if (error) {
            console.error("Hybrid search error:", error);
            return { isError: true, content: [{ type: "text", text: `Search failed: ${error.message}` }] };
          }

          return {
            content: [{
              type: "text",
              text: JSON.stringify({
                mode: queryEmbedding ? "hybrid_vector_fts" : "fts_fallback_no_embedding",
                query,
                collection: collection || "all",
                total_results: data?.length || 0,
                results: data || [],
              }, null, 2),
            }],
          };
        } else {
          // Pure vector search
          if (!queryEmbedding) {
            return {
              isError: true,
              content: [{ type: "text", text: "Vector search unavailable: failed to generate embedding for query." }],
            };
          }

          const { data, error } = await supabase.rpc("match_document_chunks", {
            query_embedding: queryEmbedding,
            match_threshold: match_threshold,
            match_count: limit,
            filter_collection: collection || null,
          });

          if (error) {
            return { isError: true, content: [{ type: "text", text: `Search failed: ${error.message}` }] };
          }

          return {
            content: [{
              type: "text",
              text: JSON.stringify({
                mode: "pure_vector",
                query,
                collection: collection || "all",
                total_results: data?.length || 0,
                results: data || [],
              }, null, 2),
            }],
          };
        }
      }

      case "rag_ingest_document": {
        const {
          title,
          content,
          collection = "default",
          source_uri = null,
          chunk_size = 1000,
          chunk_overlap = 150,
          metadata = {},
        } = args;

        if (!title || !content) {
          return { isError: true, content: [{ type: "text", text: "Arguments 'title' and 'content' are required." }] };
        }

        const docHash = await sha256Hex(content);

        // Check if document already exists with identical content
        const { data: existingDoc } = await supabase
          .from("documents")
          .select("id, title, collection, total_chunks")
          .eq("collection", collection)
          .eq("content_hash", docHash)
          .limit(1);

        if (existingDoc && existingDoc.length > 0) {
          return {
            content: [{
              type: "text",
              text: JSON.stringify({
                status: "skipped_unchanged",
                message: "Document already indexed with identical content hash.",
                document_id: existingDoc[0].id,
                total_chunks: existingDoc[0].total_chunks,
              }, null, 2),
            }],
          };
        }

        // Split content into chunks
        const rawChunks = splitTextIntoChunks(content, chunk_size, chunk_overlap);
        if (rawChunks.length === 0) {
          return { isError: true, content: [{ type: "text", text: "Document content yielded zero chunks." }] };
        }

        // Compute hashes for each chunk
        const chunkHashes = await Promise.all(rawChunks.map((c) => sha256Hex(c)));

        // Check for existing embeddings across all chunks to conserve API quota
        const { data: cachedChunks } = await supabase
          .from("document_chunks")
          .select("content_hash, embedding")
          .in("content_hash", chunkHashes)
          .not("embedding", "is", null);

        const embeddingCache = new Map<string, number[]>();
        if (cachedChunks) {
          for (const item of cachedChunks) {
            if (item.content_hash && item.embedding) {
              embeddingCache.set(item.content_hash, item.embedding);
            }
          }
        }

        // Filter chunks that require API embedding
        const chunksToEmbedIndices: number[] = [];
        const chunksToEmbedTexts: string[] = [];

        rawChunks.forEach((chunkText, idx) => {
          const hash = chunkHashes[idx];
          if (!embeddingCache.has(hash)) {
            chunksToEmbedIndices.push(idx);
            chunksToEmbedTexts.push(chunkText);
          }
        });

        // Batch embed only uncached chunks
        const newlyEmbedded: Array<number[] | null> = new Array(rawChunks.length).fill(null);
        let quotaWarning = false;

        if (chunksToEmbedTexts.length > 0) {
          // Process in sub-batches of 25 to respect free tier request size limits
          const batchSize = 25;
          for (let i = 0; i < chunksToEmbedTexts.length; i += batchSize) {
            const sliceTexts = chunksToEmbedTexts.slice(i, i + batchSize);
            const sliceIndices = chunksToEmbedIndices.slice(i, i + batchSize);
            const embedResults = await generateEmbeddingsBatch(sliceTexts);

            embedResults.forEach((res, rIdx) => {
              const origIdx = sliceIndices[rIdx];
              if (res.embedding) {
                newlyEmbedded[origIdx] = res.embedding;
                embeddingCache.set(chunkHashes[origIdx], res.embedding);
              } else if (res.status.includes("429")) {
                quotaWarning = true;
              }
            });
          }
        }

        // Upsert/Insert document record
        let docId: string;
        // Check if updating document by source_uri or title in collection
        let existingToUpdate: any = null;
        if (source_uri) {
          const { data } = await supabase
            .from("documents")
            .select("id")
            .eq("collection", collection)
            .eq("source_uri", source_uri)
            .limit(1);
          if (data && data.length > 0) existingToUpdate = data[0];
        }

        const estTokens = Math.round(content.length / 4);

        if (existingToUpdate) {
          docId = existingToUpdate.id;
          await supabase
            .from("documents")
            .update({
              title,
              content_hash: docHash,
              total_chunks: rawChunks.length,
              total_tokens: estTokens,
              metadata,
            })
            .eq("id", docId);

          // Clear old chunks (cascade delete)
          await supabase.from("document_chunks").delete().eq("document_id", docId);
        } else {
          const { data: newDoc, error: docErr } = await supabase
            .from("documents")
            .insert({
              collection,
              title,
              source_uri,
              content_hash: docHash,
              total_chunks: rawChunks.length,
              total_tokens: estTokens,
              metadata,
            })
            .select("id")
            .single();

          if (docErr || !newDoc) {
            return { isError: true, content: [{ type: "text", text: `Failed to insert document: ${docErr?.message}` }] };
          }
          docId = newDoc.id;
        }

        // Prepare chunk records
        const chunkRows = rawChunks.map((chunkText, idx) => {
          const hash = chunkHashes[idx];
          const emb = embeddingCache.get(hash) || newlyEmbedded[idx] || null;
          return {
            document_id: docId,
            chunk_index: idx,
            content: chunkText,
            content_hash: hash,
            token_count: Math.round(chunkText.length / 4),
            metadata: { ...metadata, chunk_index: idx },
            embedding: emb,
          };
        });

        // Insert chunks in batches of 50
        const insertBatchSize = 50;
        for (let i = 0; i < chunkRows.length; i += insertBatchSize) {
          const batch = chunkRows.slice(i, i + insertBatchSize);
          const { error: chunkErr } = await supabase.from("document_chunks").insert(batch);
          if (chunkErr) {
            console.error("Chunk insertion error:", chunkErr);
          }
        }

        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              status: "success",
              document_id: docId,
              title,
              collection,
              chunks_total: rawChunks.length,
              chunks_embedded_new: chunksToEmbedTexts.length,
              chunks_reused_from_cache: rawChunks.length - chunksToEmbedTexts.length,
              quota_warning_429: quotaWarning,
              estimated_tokens: estTokens,
            }, null, 2),
          }],
        };
      }

      case "rag_list_documents": {
        const { collection = null, limit = 25 } = args;
        let q = supabase
          .from("documents")
          .select("id, title, collection, source_uri, total_chunks, total_tokens, created_at, updated_at")
          .order("created_at", { ascending: false })
          .limit(limit);

        if (collection) {
          q = q.eq("collection", collection);
        }

        const { data, error } = await q;
        if (error) {
          return { isError: true, content: [{ type: "text", text: `Error: ${error.message}` }] };
        }

        return {
          content: [{
            type: "text",
            text: JSON.stringify({ count: data?.length || 0, documents: data || [] }, null, 2),
          }],
        };
      }

      case "rag_get_document": {
        const { document_id, title, include_chunks = true } = args;
        if (!document_id && !title) {
          return { isError: true, content: [{ type: "text", text: "Provide either 'document_id' or 'title'." }] };
        }

        let docQuery = supabase.from("documents").select("*");
        if (document_id) {
          docQuery = docQuery.eq("id", document_id);
        } else {
          docQuery = docQuery.ilike("title", title);
        }

        const { data: docData, error: docErr } = await docQuery.limit(1);
        if (docErr || !docData || docData.length === 0) {
          return { isError: true, content: [{ type: "text", text: "Document not found." }] };
        }

        const doc = docData[0];
        let chunks: any[] = [];

        if (include_chunks) {
          const { data: chunkData } = await supabase
            .from("document_chunks")
            .select("id, chunk_index, content, token_count, metadata, embedding")
            .eq("document_id", doc.id)
            .order("chunk_index", { ascending: true });

          chunks = (chunkData || []).map((c) => ({
            id: c.id,
            chunk_index: c.chunk_index,
            token_count: c.token_count,
            content: c.content,
            has_embedding: Boolean(c.embedding),
            metadata: c.metadata,
          }));
        }

        return {
          content: [{
            type: "text",
            text: JSON.stringify({ document: doc, chunks_count: chunks.length, chunks }, null, 2),
          }],
        };
      }

      case "rag_delete_document": {
        const { document_id, source_uri } = args;
        if (!document_id && !source_uri) {
          return { isError: true, content: [{ type: "text", text: "Provide either 'document_id' or 'source_uri'." }] };
        }

        let deleteQuery = supabase.from("documents").delete();
        if (document_id) {
          deleteQuery = deleteQuery.eq("id", document_id);
        } else {
          deleteQuery = deleteQuery.eq("source_uri", source_uri);
        }

        const { error } = await deleteQuery;
        if (error) {
          return { isError: true, content: [{ type: "text", text: `Delete failed: ${error.message}` }] };
        }

        return {
          content: [{
            type: "text",
            text: JSON.stringify({ success: true, message: "Document and associated chunks deleted." }),
          }],
        };
      }

      case "rag_storage_stats": {
        const { data, error } = await supabase.rpc("get_rag_stats");
        if (error) {
          return { isError: true, content: [{ type: "text", text: `Failed to fetch stats: ${error.message}` }] };
        }

        return {
          content: [{
            type: "text",
            text: JSON.stringify(data, null, 2),
          }],
        };
      }

      default:
        return { isError: true, content: [{ type: "text", text: `Unknown tool: ${name}` }] };
    }
  } catch (err: any) {
    return { isError: true, content: [{ type: "text", text: `Tool execution error: ${err.message || String(err)}` }] };
  }
}

/**
 * Handle incoming JSON-RPC 2.0 messages from MCP client
 */
async function handleJsonRpc(msg: any, supabase: SupabaseClient): Promise<any | null> {
  const { jsonrpc, id, method, params } = msg;

  if (method === "initialize") {
    return {
      jsonrpc: "2.0",
      id,
      result: {
        protocolVersion: "2024-11-05",
        capabilities: {
          tools: { listChanged: false },
        },
        serverInfo: {
          name: "RaggaeServer",
          version: "1.0.0",
        },
      },
    };
  }

  if (method === "notifications/initialized") {
    return null;
  }

  if (method === "ping") {
    return { jsonrpc: "2.0", id, result: {} };
  }

  if (method === "tools/list") {
    return {
      jsonrpc: "2.0",
      id,
      result: {
        tools: MCP_TOOLS,
      },
    };
  }

  if (method === "tools/call") {
    const toolName = params?.name;
    const toolArgs = params?.arguments || {};
    const toolResult = await executeTool(toolName, toolArgs, supabase);
    return {
      jsonrpc: "2.0",
      id,
      result: toolResult,
    };
  }

  if (id !== undefined) {
    return {
      jsonrpc: "2.0",
      id,
      error: {
        code: -32601,
        message: `Method '${method}' not found.`,
      },
    };
  }

  return null;
}

// Main Edge Function HTTP Handler
Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  const url = new URL(req.url);
  const pathname = url.pathname;

  const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
  const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") || Deno.env.get("SUPABASE_ANON_KEY")!;
  const supabase = createClient(supabaseUrl, supabaseKey);

  // Healthcheck endpoint
  if (req.method === "GET" && (pathname.endsWith("/health") || pathname === "/functions/v1/mcp")) {
    const geminiKey = Deno.env.get("GEMINI_API_KEY") || Deno.env.get("GOOGLE_API_KEY");
    const openaiKey = Deno.env.get("OPENAI_API_KEY");
    let testEmbeddingStatus = "not_tested";

    if (url.searchParams.get("test") === "embedding") {
      const emb = await generateSingleEmbedding("health check ping");
      testEmbeddingStatus = emb ? `ok (${emb.length} dims)` : "failed";
    }

    return new Response(
      JSON.stringify({
        status: "healthy",
        server: "RaggaeServer",
        version: "1.0.0",
        runtime: "supabase-edge",
        features: ["hybrid_retrieval", "vector_search", "full_text_search", "free_tier_protection"],
        config: {
          geminiKeyConfigured: Boolean(geminiKey),
          openaiKeyConfigured: Boolean(openaiKey),
        },
        testEmbedding: testEmbeddingStatus,
      }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }

  // Authentication: Verify Supabase API Key (service_role or anon key)
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  const anonKey = Deno.env.get("SUPABASE_ANON_KEY");
  const validKeys = [serviceKey, anonKey].filter(Boolean) as string[];

  if (validKeys.length > 0) {
    const authHeader = req.headers.get("authorization")?.replace(/^Bearer\s+/i, "");
    const apiKeyHeader = req.headers.get("apikey");
    const queryKey = url.searchParams.get("apikey") || url.searchParams.get("token") || url.searchParams.get("key");

    const providedKey = authHeader || apiKeyHeader || queryKey;
    if (!providedKey || !validKeys.includes(providedKey)) {
      return new Response(
        JSON.stringify({ error: "Unauthorized: Missing or invalid Supabase API Key" }),
        { status: 401, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }
  }

  // 1. GET /sse -> Establish Server-Sent Events stream
  if (req.method === "GET" && (pathname.endsWith("/sse") || url.searchParams.get("transport") === "sse")) {
    const sessionId = crypto.randomUUID();
    const encoder = new TextEncoder();
    let streamController: ReadableStreamDefaultController<Uint8Array>;

    const base = Deno.env.get("SUPABASE_URL") || `${url.protocol}//${url.host}`;
    const queryKey = url.searchParams.get("apikey") || url.searchParams.get("token") || url.searchParams.get("key");
    const authQuery = queryKey ? `&apikey=${encodeURIComponent(queryKey)}` : "";
    const endpointUri = `${base}/functions/v1/mcp/message?sessionId=${sessionId}${authQuery}`;

    const readable = new ReadableStream<Uint8Array>({
      start(controller) {
        streamController = controller;
        controller.enqueue(encoder.encode(`event: endpoint\ndata: ${endpointUri}\n\n`));
      },
      cancel() {
        sessions.delete(sessionId);
      },
    });

    sessions.set(sessionId, {
      send: (data: any) => {
        try {
          const payload = `event: message\ndata: ${JSON.stringify(data)}\n\n`;
          streamController.enqueue(encoder.encode(payload));
        } catch (e) {
          console.error("SSE send error:", e);
        }
      },
      close: () => {
        sessions.delete(sessionId);
      },
    });

    return new Response(readable, {
      headers: {
        ...corsHeaders,
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }

  // 2. POST /message?sessionId=... -> Deliver message to active SSE session
  if (req.method === "POST" && pathname.includes("/message")) {
    const sessionId = url.searchParams.get("sessionId");
    if (!sessionId || !sessions.has(sessionId)) {
      return new Response(
        JSON.stringify({ error: "Session not found or expired" }),
        { status: 404, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const session = sessions.get(sessionId)!;
    const msg = await req.json();

    // Process asynchronously and stream back on SSE channel
    (async () => {
      const resp = await handleJsonRpc(msg, supabase);
      if (resp) session.send(resp);
    })();

    return new Response(JSON.stringify({ status: "accepted" }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }

  // 3. POST / -> Standard HTTP Streamable JSON-RPC Endpoint (Direct MCP Request/Response)
  if (req.method === "POST") {
    try {
      const msg = await req.json();
      const resp = await handleJsonRpc(msg, supabase);
      return new Response(JSON.stringify(resp || {}), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    } catch (err: any) {
      return new Response(
        JSON.stringify({ jsonrpc: "2.0", id: null, error: { code: -32700, message: "Parse error" } }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }
  }

  return new Response("Not Found", { status: 404, headers: corsHeaders });
});
