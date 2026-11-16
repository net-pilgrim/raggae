---
name: raggae
description: >-
  Guide and best practices for querying, ingesting, and managing persistent documents
  in the free-tier serverless Raggae RAG engine on Supabase Edge & pgvector.
---

# Raggae: Serverless Free-Tier Hybrid RAG Engine

Raggae is a 100% serverless, zero-cost RAG (Retrieval-Augmented Generation) system hosted on Supabase Edge Functions with `pgvector` HNSW indexes and PostgreSQL Full-Text Search (GIN).

---

## 🎯 When to Call RAG Tools

Agent Goal | Tool to Call | Key Parameters
:--- | :--- | :---
**Find relevant context or code** | `rag_search` | `query="..."`, `collection="docs"`, `hybrid=true`, `limit=8`
**Ingest a new guide/document** | `rag_ingest_document` | `title="..."`, `content="..."`, `collection="docs"`, `source_uri="..."`
**Inspect available knowledge** | `rag_list_documents` | `collection="..."`, `limit=25`
**Read an entire indexed doc** | `rag_get_document` | `document_id="..."` or `title="..."`, `include_chunks=true`
**Remove outdated documentation** | `rag_delete_document` | `document_id="..."` or `source_uri="..."`
**Check free-tier storage usage** | `rag_storage_stats` | *(no parameters)*

---

## 🔍 Search & Retrieval Examples

### 1. Hybrid Search (Recommended)
Combines dense vector similarity (Gemini 768-dim) and English keyword full-text search using Reciprocal Rank Fusion (RRF):

```json
{
  "name": "rag_search",
  "arguments": {
    "query": "How to configure Supabase edge function rate limits and CORS",
    "collection": "guides",
    "limit": 6,
    "hybrid": true
  }
}
```

### 2. Scoped Collection Search
Isolate queries to a specific project, repository, or namespace:

```json
{
  "name": "rag_search",
  "arguments": {
    "query": "JWT authentication middleware implementation",
    "collection": "codebase",
    "limit": 5
  }
}
```

---

## 📥 Ingesting Knowledge

When persisting long manuals, architecture summaries, or reference specs:

```json
{
  "name": "rag_ingest_document",
  "arguments": {
    "title": "Database Migration Runbook v2",
    "collection": "runbooks",
    "source_uri": "docs/devops/migration_runbook.md",
    "content": "# Database Migration Guide\n\n1. Ensure pgvector extension...",
    "chunk_size": 1000,
    "chunk_overlap": 150,
    "metadata": {
      "author": "devops_agent",
      "version": "2.1",
      "tags": ["database", "migrations", "supabase"]
    }
  }
}
```

> [!TIP]
> Raggae computes SHA-256 content hashes for both documents and individual chunks. Re-ingesting unchanged content automatically reuses existing vector embeddings with **zero API consumption**.

---

## 📊 Staying Within the Supabase 500MB Free Tier

Always call `rag_storage_stats` when performing bulk operations to verify database capacity:

```json
{
  "name": "rag_storage_stats",
  "arguments": {}
}
```

Key Metrics:
- **`total_storage_mb`**: Current database size (free tier cap: 500.0 MB).
- **`free_tier_used_pct`**: Percentage of free database capacity consumed.
- **`estimated_capacity_remaining_chunks`**: Remaining safe headroom for new text chunks.
