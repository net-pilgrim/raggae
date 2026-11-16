#!/usr/bin/env node
/**
 * Raggae CLI - Free-Tier Rate-Controlled RAG Ingestion & Query Tool
 * Built for Supabase (pgvector) + Google Gemini (text-embedding-004)
 */

import { promises as fs } from "node:fs";
import * as path from "node:path";
import * as crypto from "node:crypto";
import { performance } from "node:perf_hooks";

// --- Simple .env Loader ---
async function loadEnv() {
  try {
    const envPath = path.resolve(process.cwd(), ".env");
    const content = await fs.readFile(envPath, "utf-8");
    for (const line of content.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith("#")) continue;
      const eqIdx = trimmed.indexOf("=");
      if (eqIdx !== -1) {
        const key = trimmed.slice(0, eqIdx).trim();
        let val = trimmed.slice(eqIdx + 1).trim();
        if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
          val = val.slice(1, -1);
        }
        if (!process.env[key]) {
          process.env[key] = val;
        }
      }
    }
  } catch (_) {
    // No .env file, rely on environment variables
  }
}

// --- Token Bucket Rate Limiter ---
class TokenBucketLimiter {
  constructor(tps = 1.0, maxBurst = 2.0) {
    this.capacity = maxBurst;
    this.tokens = maxBurst;
    this.fillRate = tps; // tokens per second
    this.lastCheck = performance.now();
  }

  async acquire(cost = 1.0) {
    while (true) {
      const now = performance.now();
      const elapsedSec = (now - this.lastCheck) / 1000.0;
      this.lastCheck = now;
      this.tokens = Math.min(this.capacity, this.tokens + elapsedSec * this.fillRate);

      if (this.tokens >= cost) {
        this.tokens -= cost;
        return;
      }

      // Wait until enough tokens are replenished
      const needed = cost - this.tokens;
      const waitMs = Math.ceil((needed / this.fillRate) * 1000);
      await new Promise((resolve) => setTimeout(resolve, Math.max(waitMs, 50)));
    }
  }
}

// --- Helper Functions ---
function sha256(text) {
  const clean = text.replace(/\s+/g, " ").trim();
  return crypto.createHash("sha256").update(clean).digest("hex");
}

function splitTextIntoChunks(text, chunkSize = 1000, chunkOverlap = 150) {
  if (!text || text.trim().length === 0) return [];
  if (text.length <= chunkSize) return [text.trim()];

  const separators = ["\n\n", "\n", "```", ". ", "; ", " ", ""];

  function split(textToSplit, sepIndex) {
    if (textToSplit.length <= chunkSize || sepIndex >= separators.length) {
      return [textToSplit.trim()].filter(Boolean);
    }

    const sep = separators[sepIndex];
    const rawParts = sep ? textToSplit.split(sep) : Array.from(textToSplit);
    const result = [];
    let current = "";

    for (const part of rawParts) {
      const candidate = current ? current + sep + part : part;
      if (candidate.length <= chunkSize) {
        current = candidate;
      } else {
        if (current) result.push(current.trim());
        if (part.length > chunkSize) {
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
  if (rawChunks.length <= 1 || chunkOverlap <= 0) {
    return rawChunks;
  }

  const overlapped = [];
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

async function findFilesRecursively(dirPath, extensions = null) {
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  let files = [];

  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    if (entry.name.startsWith(".") || entry.name === "node_modules" || entry.name === "dist") {
      continue;
    }

    if (entry.isDirectory()) {
      const subFiles = await findFilesRecursively(fullPath, extensions);
      files.push(...subFiles);
    } else if (entry.isFile()) {
      if (!extensions) {
        files.push(fullPath);
      } else {
        const ext = path.extname(entry.name).toLowerCase();
        if (extensions.includes(ext)) {
          files.push(fullPath);
        }
      }
    }
  }

  return files;
}

// --- Gemini Embeddings with Rate Control & Exponential Backoff ---
async function batchEmbedGemini(texts, geminiKey, limiter) {
  if (texts.length === 0) return [];

  let attempts = 0;
  const maxAttempts = 5;
  let backoffMs = 1500;

  while (attempts < maxAttempts) {
    // Acquire rate limit token before network call
    await limiter.acquire(1.0);

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
        const data = await res.json();
        const embs = data.embeddings || [];
        return texts.map((_, i) => embs[i]?.values || null);
      }

      if (res.status === 429) {
        attempts++;
        const retryAfter = res.headers.get("retry-after");
        const sleepDuration = retryAfter ? parseInt(retryAfter, 10) * 1000 : backoffMs;
        console.warn(`⚠️  Gemini 429 Rate Limit (Free Tier). Backing off for ${Math.round(sleepDuration / 1000)}s (attempt ${attempts}/${maxAttempts})...`);
        await new Promise((r) => setTimeout(r, sleepDuration));
        backoffMs *= 2;
        continue;
      }

      const errText = await res.text();
      console.error(`Gemini embedding error (${res.status}):`, errText);
      return texts.map(() => null);
    } catch (err) {
      attempts++;
      console.warn(`Network glitch: ${err.message}. Retrying in ${backoffMs}ms...`);
      await new Promise((r) => setTimeout(r, backoffMs));
      backoffMs *= 2;
    }
  }

  return texts.map(() => null);
}

// --- Supabase Client (Zero external dependency fetch client) ---
class SupabaseHttp {
  constructor(url, key, limiter) {
    this.url = url.replace(/\/$/, "");
    this.key = key;
    this.limiter = limiter;
  }

  async request(endpoint, options = {}) {
    await this.limiter.acquire(0.5); // Database operations token
    const headers = {
      apikey: this.key,
      Authorization: `Bearer ${this.key}`,
      "Content-Type": "application/json",
      Prefer: "return=representation",
      ...(options.headers || {}),
    };

    const fullUrl = `${this.url}/rest/v1/${endpoint}`;
    let res = await fetch(fullUrl, { ...options, headers });

    if (res.status === 429) {
      console.warn("⚠️  Supabase Rate Limit hit. Backing off 2s...");
      await new Promise((r) => setTimeout(r, 2000));
      res = await fetch(fullUrl, { ...options, headers });
    }

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Supabase request failed (${res.status}): ${err}`);
    }

    const text = await res.text();
    return text ? JSON.parse(text) : null;
  }

  async rpc(fnName, params = {}) {
    await this.limiter.acquire(0.5);
    const headers = {
      apikey: this.key,
      Authorization: `Bearer ${this.key}`,
      "Content-Type": "application/json",
    };

    const fullUrl = `${this.url}/rest/v1/rpc/${fnName}`;
    const res = await fetch(fullUrl, {
      method: "POST",
      headers,
      body: JSON.stringify(params),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Supabase RPC ${fnName} failed (${res.status}): ${err}`);
    }

    return await res.json();
  }
}

// --- CLI Commands ---

async function cmdIngest(targetPath, flags) {
  const isDryRun = Boolean(flags["dry-run"]);
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY;
  const geminiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;

  if (!isDryRun && (!supabaseUrl || !supabaseKey)) {
    console.error("❌ Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env or environment.");
    process.exit(1);
  }

  if (!geminiKey && !flags["dry-run"]) {
    console.warn("⚠️  Warning: GEMINI_API_KEY is not set. Chunks will be stored without vector embeddings (FTS keyword search only).");
  }

  const tps = parseFloat(flags.tps || process.env.RAGGAE_TPS || "1.5");
  const batchSize = parseInt(flags["batch-size"] || process.env.RAGGAE_BATCH_SIZE || "25", 10);
  const chunkSize = parseInt(flags["chunk-size"] || "1000", 10);
  const chunkOverlap = parseInt(flags["chunk-overlap"] || "150", 10);
  const collection = flags.collection || "default";

  console.log(`\n🚀 Raggae Free-Tier Ingestion Engine`);
  console.log(`--------------------------------------------------`);
  console.log(`Target:       ${targetPath}`);
  console.log(`Collection:   ${collection}`);
  console.log(`Rate Control: ${tps} TPS (Token Bucket)`);
  console.log(`Batch Size:   ${batchSize} chunks/req`);
  console.log(`Chunk Size:   ${chunkSize} chars (overlap: ${chunkOverlap})`);
  console.log(`--------------------------------------------------\n`);

  const limiter = new TokenBucketLimiter(tps, tps * 2);
  const db = !isDryRun ? new SupabaseHttp(supabaseUrl, supabaseKey, limiter) : null;

  // Discover files
  const stat = await fs.stat(targetPath);
  let filePaths = [];
  const validExts = [".md", ".txt", ".json", ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".csv", ".sql", ".sh"];

  if (stat.isDirectory()) {
    filePaths = await findFilesRecursively(targetPath, validExts);
  } else {
    filePaths = [targetPath];
  }

  console.log(`Found ${filePaths.length} file(s) to process.\n`);

  let totalDocsProcessed = 0;
  let totalDocsSkipped = 0;
  let totalChunksIngested = 0;
  let totalChunksEmbedded = 0;
  let totalChunksReused = 0;
  const startTime = performance.now();

  for (let i = 0; i < filePaths.length; i++) {
    const filePath = filePaths[i];
    const relPath = path.relative(process.cwd(), filePath);
    const content = await fs.readFile(filePath, "utf-8");

    if (!content.trim()) continue;

    const docHash = sha256(content);
    const title = path.basename(filePath);

    process.stdout.write(`[${i + 1}/${filePaths.length}] ${relPath} ... `);

    // 1. Check if document already exists with identical content
    if (!isDryRun && !flags["force"]) {
      const existing = await db.request(`documents?collection=eq.${encodeURIComponent(collection)}&content_hash=eq.${docHash}&select=id,total_chunks`);
      if (existing && existing.length > 0) {
        process.stdout.write(`⏭️  Skipped (unchanged)\n`);
        totalDocsSkipped++;
        continue;
      }
    }

    // 2. Chunk content
    const chunks = splitTextIntoChunks(content, chunkSize, chunkOverlap);
    if (chunks.length === 0) {
      process.stdout.write(`⚠️  Empty chunks\n`);
      continue;
    }

    if (flags["dry-run"]) {
      process.stdout.write(`🔍 Dry Run: ${chunks.length} chunks (~${Math.round(content.length / 4)} tokens)\n`);
      continue;
    }

    // 3. Compute chunk hashes
    const chunkHashes = chunks.map((c) => sha256(c));

    // 4. Check embedding cache in DB (reuse existing vectors to stay well below API limits!)
    const cachedChunks = await db.request(
      `document_chunks?content_hash=in.(${chunkHashes.map((h) => `"${h}"`).join(",")})&embedding=not.is.null&select=content_hash,embedding`
    );

    const embeddingCache = new Map();
    if (cachedChunks) {
      for (const item of cachedChunks) {
        if (item.content_hash && item.embedding) {
          embeddingCache.set(item.content_hash, item.embedding);
        }
      }
    }

    // 5. Embed uncached chunks in rate-controlled batches
    const uncachedIndices = [];
    const uncachedTexts = [];
    chunks.forEach((chunkText, idx) => {
      const hash = chunkHashes[idx];
      if (!embeddingCache.has(hash)) {
        uncachedIndices.push(idx);
        uncachedTexts.push(chunkText);
      }
    });

    const newlyEmbedded = new Array(chunks.length).fill(null);

    if (geminiKey && uncachedTexts.length > 0) {
      for (let b = 0; b < uncachedTexts.length; b += batchSize) {
        const sliceTexts = uncachedTexts.slice(b, b + batchSize);
        const sliceIndices = uncachedIndices.slice(b, b + batchSize);
        const embeddings = await batchEmbedGemini(sliceTexts, geminiKey, limiter);

        embeddings.forEach((emb, eIdx) => {
          if (emb) {
            newlyEmbedded[sliceIndices[eIdx]] = emb;
            embeddingCache.set(chunkHashes[sliceIndices[eIdx]], emb);
            totalChunksEmbedded++;
          }
        });
      }
    }

    totalChunksReused += chunks.length - uncachedTexts.length;

    // 6. Upsert document in database
    const existingDoc = await db.request(`documents?collection=eq.${encodeURIComponent(collection)}&source_uri=eq.${encodeURIComponent(relPath)}&select=id`);
    let docId;

    const estTokens = Math.round(content.length / 4);

    if (existingDoc && existingDoc.length > 0) {
      docId = existingDoc[0].id;
      await db.request(`documents?id=eq.${docId}`, {
        method: "PATCH",
        body: JSON.stringify({
          title,
          content_hash: docHash,
          total_chunks: chunks.length,
          total_tokens: estTokens,
          metadata: { file_ext: path.extname(filePath), size_bytes: content.length },
        }),
      });
      // Delete old chunks
      await db.request(`document_chunks?document_id=eq.${docId}`, { method: "DELETE" });
    } else {
      const inserted = await db.request(`documents`, {
        method: "POST",
        body: JSON.stringify({
          collection,
          title,
          source_uri: relPath,
          content_hash: docHash,
          total_chunks: chunks.length,
          total_tokens: estTokens,
          metadata: { file_ext: path.extname(filePath), size_bytes: content.length },
        }),
      });
      docId = inserted[0]?.id;
    }

    // 7. Insert chunk rows in DB
    const chunkRows = chunks.map((chunkText, idx) => {
      const hash = chunkHashes[idx];
      const emb = embeddingCache.get(hash) || newlyEmbedded[idx] || null;
      return {
        document_id: docId,
        chunk_index: idx,
        content: chunkText,
        content_hash: hash,
        token_count: Math.round(chunkText.length / 4),
        metadata: { chunk_index: idx, source_file: relPath },
        embedding: emb,
      };
    });

    // Sub-batch DB insertion to avoid hitting request payload caps
    for (let c = 0; c < chunkRows.length; c += 50) {
      const batchRows = chunkRows.slice(c, c + 50);
      await db.request(`document_chunks`, {
        method: "POST",
        body: JSON.stringify(batchRows),
      });
    }

    totalChunksIngested += chunks.length;
    totalDocsProcessed++;
    process.stdout.write(`✅ Indexed (${chunks.length} chunks, ${chunks.length - uncachedTexts.length} cached)\n`);
  }

  const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
  console.log(`\n==================================================`);
  console.log(`🎉 Ingestion Summary:`);
  console.log(`Documents Processed: ${totalDocsProcessed}`);
  console.log(`Documents Skipped:   ${totalDocsSkipped} (unchanged)`);
  console.log(`Chunks Ingested:     ${totalChunksIngested}`);
  console.log(`New Embeddings:      ${totalChunksEmbedded}`);
  console.log(`Vectors Reused:      ${totalChunksReused} (zero API cost)`);
  console.log(`Total Time Elapsed:  ${elapsed}s`);
  console.log(`==================================================\n`);
}

async function cmdSearch(query, flags) {
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY;
  const geminiKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;

  if (!supabaseUrl || !supabaseKey) {
    console.error("❌ Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.");
    process.exit(1);
  }

  const limiter = new TokenBucketLimiter(5.0, 5.0);
  const db = new SupabaseHttp(supabaseUrl, supabaseKey, limiter);
  const limit = parseInt(flags.limit || "5", 10);
  const collection = flags.collection || null;

  console.log(`\n🔎 Query: "${query}"`);
  console.log(`Collection: ${collection || "all"} | Limit: ${limit}\n`);

  let queryEmbedding = null;
  if (geminiKey) {
    const embs = await batchEmbedGemini([query], geminiKey, limiter);
    queryEmbedding = embs[0];
  }

  const results = await db.rpc("match_document_chunks_hybrid", {
    query_text: query,
    query_embedding: queryEmbedding,
    match_count: limit,
    filter_collection: collection,
  });

  if (!results || results.length === 0) {
    console.log("No matching documents found.");
    return;
  }

  results.forEach((r, idx) => {
    console.log(`[Result #${idx + 1}]`);
    console.log(`Title:      ${r.doc_title} (${r.doc_collection})`);
    console.log(`Source:     ${r.doc_source_uri || "n/a"}`);
    console.log(`Score:      ${r.score?.toFixed ? r.score.toFixed(4) : r.score} | Sim: ${r.similarity?.toFixed ? r.similarity.toFixed(4) : r.similarity}`);
    console.log(`Content:    ${r.content.slice(0, 200).replace(/\n/g, " ")}...`);
    console.log(`--------------------------------------------------`);
  });
}

async function cmdStats() {
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY;

  if (!supabaseUrl || !supabaseKey) {
    console.error("❌ Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.");
    process.exit(1);
  }

  const limiter = new TokenBucketLimiter(5.0, 5.0);
  const db = new SupabaseHttp(supabaseUrl, supabaseKey, limiter);

  try {
    const stats = await db.rpc("get_rag_stats");
    console.log(`\n📊 Raggae Supabase Free-Tier Storage Report`);
    console.log(`==================================================`);
    console.log(`Documents Count:            ${stats.documents_count}`);
    console.log(`Total Chunks:               ${stats.chunks_count}`);
    console.log(`Embedded Chunks:            ${stats.embedded_chunks_count}`);
    console.log(`Collections:                ${stats.distinct_collections}`);
    console.log(`--------------------------------------------------`);
    console.log(`Total DB Storage Used:      ${stats.total_storage_mb} MB / ${stats.free_tier_limit_mb} MB`);
    console.log(`Table Data:                 ${stats.table_storage_mb} MB`);
    console.log(`Vector & FTS Indexes:       ${stats.index_storage_mb} MB`);
    console.log(`Free Tier Utilization:      ${stats.free_tier_used_pct}%`);
    console.log(`Headroom Remaining Chunks:  ~${stats.estimated_capacity_remaining_chunks} chunks`);
    console.log(`==================================================\n`);
  } catch (err) {
    console.error("Failed to query stats:", err.message);
  }
}

async function cmdList(flags) {
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_KEY;

  if (!supabaseUrl || !supabaseKey) {
    console.error("❌ Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.");
    process.exit(1);
  }

  const limiter = new TokenBucketLimiter(5.0, 5.0);
  const db = new SupabaseHttp(supabaseUrl, supabaseKey, limiter);
  const collection = flags.collection;

  let query = "documents?select=id,title,collection,source_uri,total_chunks,total_tokens,created_at&order=created_at.desc&limit=50";
  if (collection) {
    query += `&collection=eq.${encodeURIComponent(collection)}`;
  }

  const docs = await db.request(query);
  console.log(`\n📚 Indexed Documents (${docs.length}):`);
  console.log(`--------------------------------------------------------------------------------`);
  console.log(`Title                     | Collection | Chunks | Tokens | Source`);
  console.log(`--------------------------------------------------------------------------------`);
  docs.forEach((d) => {
    const title = (d.title || "untitled").slice(0, 25).padEnd(25);
    const col = (d.collection || "default").slice(0, 10).padEnd(10);
    const chunks = String(d.total_chunks || 0).padStart(6);
    const tokens = String(d.total_tokens || 0).padStart(6);
    const src = (d.source_uri || "-").slice(0, 30);
    console.log(`${title} | ${col} | ${chunks} | ${tokens} | ${src}`);
  });
  console.log(`--------------------------------------------------------------------------------\n`);
}

function printHelp() {
  console.log(`
Raggae CLI - Serverless Free-Tier Hybrid RAG Engine on Supabase Edge & pgvector

Usage:
  raggae <command> [arguments] [flags]

Commands:
  ingest <path>          Ingest file or recursive directory with token-bucket rate limiter
  search <query>         Execute hybrid vector + keyword search over indexed documents
  stats                  Inspect Supabase 500MB free-tier storage usage and chunk headroom
  list                   List indexed documents, chunk counts, and collections
  help                   Show this help menu

Flags:
  --collection <name>    Namespace/collection partition (default: 'default')
  --tps <rate>           Target transactions per second for rate control (default: 1.5)
  --batch-size <num>     Embedding batch size per API call (default: 25)
  --chunk-size <chars>   Max characters per chunk (default: 1000)
  --chunk-overlap <num>  Overlap between consecutive chunks (default: 150)
  --limit <num>          Max results to return in search (default: 5)
  --dry-run              Inspect and chunk without sending network requests
  --force                Re-index documents even if content hash matches
  `);
}

// --- Main CLI Entrypoint ---
async function main() {
  await loadEnv();

  const args = process.argv.slice(2);
  if (args.length === 0) {
    printHelp();
    return;
  }

  const command = args[0];
  const positionalArgs = [];
  const flags = {};

  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    if (arg.startsWith("--")) {
      const key = arg.slice(2);
      if (i + 1 < args.length && !args[i + 1].startsWith("--")) {
        flags[key] = args[i + 1];
        i++;
      } else {
        flags[key] = true;
      }
    } else {
      positionalArgs.push(arg);
    }
  }

  switch (command) {
    case "ingest": {
      const target = positionalArgs[0];
      if (!target) {
        console.error("❌ Please provide a file or directory path to ingest.");
        process.exit(1);
      }
      await cmdIngest(target, flags);
      break;
    }
    case "search": {
      const query = positionalArgs.join(" ");
      if (!query) {
        console.error("❌ Please provide a query to search.");
        process.exit(1);
      }
      await cmdSearch(query, flags);
      break;
    }
    case "stats":
      await cmdStats();
      break;
    case "list":
      await cmdList(flags);
      break;
    case "help":
    case "--help":
    case "-h":
      printHelp();
      break;
    default:
      console.error(`Unknown command: ${command}`);
      printHelp();
      process.exit(1);
  }
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
