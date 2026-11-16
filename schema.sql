-- ==============================================================================
-- Supabase Schema for Free-Tier Hybrid RAG Engine (Raggae)
-- Vector & Full-Text Search Architecture optimized for Supabase Free Tier
-- ==============================================================================

-- 1. Enable required extensions
create extension if not exists vector;
create extension if not exists pg_trgm;

-- 2. Create documents table (Stores document-level metadata and tracking)
create table if not exists public.documents (
    id uuid primary key default gen_random_uuid(),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
    collection text default 'default' not null,    -- Namespace / project grouping (e.g. 'docs', 'code', 'research')
    title text not null,                          -- Document title or file name
    source_uri text,                              -- Source URL, repository path, or file path
    content_hash text not null,                   -- SHA-256 hash of raw document content to prevent re-processing
    total_chunks int default 0 not null,          -- Number of chunks belonging to this document
    total_tokens int default 0 not null,          -- Estimated token count across all chunks
    metadata jsonb default '{}'::jsonb not null   -- Arbitrary metadata (author, version, license, tags, etc.)
);

-- 3. Create document_chunks table (Stores split passages and 768-dim embeddings)
create table if not exists public.document_chunks (
    id uuid primary key default gen_random_uuid(),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    document_id uuid not null references public.documents(id) on delete cascade,
    chunk_index int not null,                     -- Zero-based sequential order of chunk in document
    content text not null,                        -- Cleaned chunk passage text
    content_hash text not null,                   -- SHA-256 hash of chunk content to reuse embeddings
    token_count int default 0 not null,           -- Approximate token count
    metadata jsonb default '{}'::jsonb not null,  -- Chunk-specific metadata (headers, section, page, etc.)
    embedding vector(768)                         -- Gemini text-embedding-004 (768 dimensions)
);

-- 4. Automatic timestamp updater trigger for documents
create or replace function update_updated_at_column()
returns trigger as $$
begin
    new.updated_at = timezone('utc'::text, now());
    return new;
end;
$$ language plpgsql;

drop trigger if exists set_documents_updated_at on public.documents;
create trigger set_documents_updated_at
    before update on public.documents
    for each row
    execute function update_updated_at_column();

-- 5. Indexes for fast retrieval and joins
create index if not exists idx_documents_collection on public.documents(collection);
create index if not exists idx_documents_source_uri on public.documents(source_uri);
create index if not exists idx_documents_content_hash on public.documents(content_hash);
create index if not exists idx_documents_metadata on public.documents using gin(metadata);
create index if not exists idx_documents_created_at on public.documents(created_at desc);

create index if not exists idx_chunks_document_id on public.document_chunks(document_id);
create index if not exists idx_chunks_content_hash on public.document_chunks(content_hash);
create index if not exists idx_chunks_metadata on public.document_chunks using gin(metadata);

-- Full-text search index on chunk content (GIN index for keyword retrieval)
create index if not exists idx_chunks_fts on public.document_chunks using gin(to_tsvector('english', content));

-- HNSW Vector index on embedding using Cosine Similarity (m=16, ef_construction=64 optimized for memory efficiency)
create index if not exists idx_chunks_embedding on public.document_chunks 
    using hnsw (embedding vector_cosine_ops)
    with (m = 16, ef_construction = 64);

-- 6. RPC function: match_document_chunks (Pure Vector Semantic Search)
create or replace function match_document_chunks (
    query_embedding vector(768),
    match_threshold float default 0.2,
    match_count int default 10,
    filter_collection text default null,
    filter_document_id uuid default null,
    filter_metadata jsonb default null
)
returns table (
    id uuid,
    document_id uuid,
    chunk_index int,
    content text,
    metadata jsonb,
    similarity float,
    doc_title text,
    doc_source_uri text,
    doc_collection text
)
language plpgsql
as $$
begin
    return query
    select
        c.id,
        c.document_id,
        c.chunk_index,
        c.content,
        c.metadata,
        1 - (c.embedding <=> query_embedding) as similarity,
        d.title as doc_title,
        d.source_uri as doc_source_uri,
        d.collection as doc_collection
    from public.document_chunks c
    join public.documents d on d.id = c.document_id
    where c.embedding is not null
      and (filter_collection is null or d.collection = filter_collection)
      and (filter_document_id is null or c.document_id = filter_document_id)
      and (filter_metadata is null or c.metadata @> filter_metadata)
      and (1 - (c.embedding <=> query_embedding)) >= match_threshold
    order by c.embedding <=> query_embedding
    limit match_count;
end;
$$;

-- 7. RPC function: search_document_chunks_fts (Full-Text Search keyword fallback)
create or replace function search_document_chunks_fts (
    query_text text,
    match_count int default 10,
    filter_collection text default null,
    filter_document_id uuid default null
)
returns table (
    id uuid,
    document_id uuid,
    chunk_index int,
    content text,
    metadata jsonb,
    score float,
    doc_title text,
    doc_source_uri text,
    doc_collection text
)
language plpgsql
as $$
begin
    return query
    select
        c.id,
        c.document_id,
        c.chunk_index,
        c.content,
        c.metadata,
        ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text))::float as score,
        d.title as doc_title,
        d.source_uri as doc_source_uri,
        d.collection as doc_collection
    from public.document_chunks c
    join public.documents d on d.id = c.document_id
    where to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
      and (filter_collection is null or d.collection = filter_collection)
      and (filter_document_id is null or c.document_id = filter_document_id)
    order by score desc
    limit match_count;
end;
$$;

-- 8. RPC function: match_document_chunks_hybrid (Hybrid Vector + Keyword Search with Reciprocal Rank Fusion)
create or replace function match_document_chunks_hybrid (
    query_text text,
    query_embedding vector(768) default null,
    match_count int default 10,
    filter_collection text default null,
    rrf_k float default 60.0,
    vector_weight float default 0.7,
    fts_weight float default 0.3
)
returns table (
    id uuid,
    document_id uuid,
    chunk_index int,
    content text,
    metadata jsonb,
    score float,
    similarity float,
    doc_title text,
    doc_source_uri text,
    doc_collection text
)
language plpgsql
as $$
declare
    has_vector boolean := (query_embedding is not null);
begin
    if not has_vector then
        -- Full-text search fallback when vector is unavailable
        return query
        select
            c.id,
            c.document_id,
            c.chunk_index,
            c.content,
            c.metadata,
            ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text))::float as score,
            0.0::float as similarity,
            d.title as doc_title,
            d.source_uri as doc_source_uri,
            d.collection as doc_collection
        from public.document_chunks c
        join public.documents d on d.id = c.document_id
        where to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
          and (filter_collection is null or d.collection = filter_collection)
        order by score desc
        limit match_count;
    else
        -- Reciprocal Rank Fusion / Combined Scoring
        return query
        with vector_matches as (
            select
                c.id,
                1 - (c.embedding <=> query_embedding) as sim,
                row_number() over (order by c.embedding <=> query_embedding) as rank
            from public.document_chunks c
            join public.documents d on d.id = c.document_id
            where c.embedding is not null
              and (filter_collection is null or d.collection = filter_collection)
            order by c.embedding <=> query_embedding
            limit match_count * 2
        ),
        fts_matches as (
            select
                c.id,
                ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text))::float as fts_score,
                row_number() over (order by ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text)) desc) as rank
            from public.document_chunks c
            join public.documents d on d.id = c.document_id
            where to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
              and (filter_collection is null or d.collection = filter_collection)
            order by fts_score desc
            limit match_count * 2
        ),
        combined as (
            select
                coalesce(v.id, f.id) as chunk_id,
                coalesce(v.sim, 0.0) as sim_val,
                (
                    coalesce(vector_weight * (1.0 / (rrf_k + v.rank)), 0.0) +
                    coalesce(fts_weight * (1.0 / (rrf_k + f.rank)), 0.0)
                ) as rrf_score
            from vector_matches v
            full outer join fts_matches f on v.id = f.id
        )
        select
            c.id,
            c.document_id,
            c.chunk_index,
            c.content,
            c.metadata,
            cb.rrf_score as score,
            cb.sim_val as similarity,
            d.title as doc_title,
            d.source_uri as doc_source_uri,
            d.collection as doc_collection
        from combined cb
        join public.document_chunks c on c.id = cb.chunk_id
        join public.documents d on d.id = c.document_id
        order by cb.rrf_score desc
        limit match_count;
    end if;
end;
$$;

-- 9. RPC function: get_rag_stats (Storage and free-tier capacity reporting)
create or replace function get_rag_stats()
returns jsonb
language plpgsql
as $$
declare
    doc_count bigint;
    chunk_count bigint;
    embedded_chunk_count bigint;
    distinct_collections bigint;
    total_db_bytes bigint;
    table_bytes bigint;
    index_bytes bigint;
    stats jsonb;
begin
    select count(*) into doc_count from public.documents;
    select count(*) into chunk_count from public.document_chunks;
    select count(*) into embedded_chunk_count from public.document_chunks where embedding is not null;
    select count(distinct collection) into distinct_collections from public.documents;

    -- Estimate table and index size in bytes
    select pg_total_relation_size('public.documents') + pg_total_relation_size('public.document_chunks')
    into total_db_bytes;

    select pg_relation_size('public.documents') + pg_relation_size('public.document_chunks')
    into table_bytes;

    index_bytes := total_db_bytes - table_bytes;

    stats := jsonb_build_object(
        'documents_count', doc_count,
        'chunks_count', chunk_count,
        'embedded_chunks_count', embedded_chunk_count,
        'distinct_collections', distinct_collections,
        'total_storage_bytes', total_db_bytes,
        'total_storage_mb', round((total_db_bytes / (1024.0 * 1024.0))::numeric, 2),
        'table_storage_mb', round((table_bytes / (1024.0 * 1024.0))::numeric, 2),
        'index_storage_mb', round((index_bytes / (1024.0 * 1024.0))::numeric, 2),
        'free_tier_limit_mb', 500.0,
        'free_tier_used_pct', round(((total_db_bytes / (500.0 * 1024.0 * 1024.0)) * 100.0)::numeric, 2),
        'estimated_capacity_remaining_chunks', greatest(0, round((500.0 * 1024.0 * 1024.0 - total_db_bytes) / 3500.0))
    );

    return stats;
end;
$$;

-- 10. Row Level Security (RLS) Policies
alter table public.documents enable row level security;
drop policy if exists "Allow service role full access on documents" on public.documents;
create policy "Allow service role full access on documents" on public.documents for all to service_role using (true) with check (true);
drop policy if exists "Allow anon full access on documents" on public.documents;
create policy "Allow anon full access on documents" on public.documents for all to anon using (true) with check (true);

alter table public.document_chunks enable row level security;
drop policy if exists "Allow service role full access on document_chunks" on public.document_chunks;
create policy "Allow service role full access on document_chunks" on public.document_chunks for all to service_role using (true) with check (true);
drop policy if exists "Allow anon full access on document_chunks" on public.document_chunks;
create policy "Allow anon full access on document_chunks" on public.document_chunks for all to anon using (true) with check (true);
