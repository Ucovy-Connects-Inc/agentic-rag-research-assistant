# System Architecture & Data Flow

This document details the architectural topology, service interactions, retrieval pipelines, exact-match caching layer, and observability framework powering the production Agentic RAG microservice.

---

## 1. High-Level Architectural Topology

The system is engineered as an enterprise-grade, microservice-based Retrieval-Augmented Generation (RAG) platform specialized for Computer Science and Artificial Intelligence research papers from arXiv.

```mermaid
flowchart TD
    subgraph ClientLayer["Client & Integration Layer"]
        Gradio["Gradio Web UI\n(Port 7861)"]
        Telegram["Telegram Bot Client\n(Long-Polling)"]
        RestClients["External REST Clients\n(Notebooks / Apps)"]
    end

    subgraph Gateway["Application Layer (FastAPI)"]
        API["FastAPI Microservice Engine\n(Port 8000)"]
        RouterAsk["/api/v1/ask\n(Standard RAG)"]
        RouterAgentic["/api/v1/agentic_ask\n(Agentic RAG)"]
        RouterStream["/api/v1/stream\n(SSE Stream)"]
        RouterHealth["/api/v1/health\n(Readiness Probe)"]
    end

    subgraph StorageLayer["Data & Indexing Tier"]
        Redis[("Redis 7 (Alpine)\nExact-Match LRU Cache\nPort 6379")]
        OpenSearch[("OpenSearch 2.19\nHybrid Index (BM25 + k-NN)\nPorts 9200 / 9600")]
        Postgres[("PostgreSQL 16\nMetadata & Relational Store\nPort 5432")]
    end

    subgraph InferenceLayer["Model & Embedding Engines"]
        Ollama["Ollama LLM Engine\n(llama3.2:1b / llama3.2:3b)\nPort 11434"]
        JinaAPI["Jina AI Embeddings API\n(1024-dim Vector Cloud)"]
    end

    subgraph ObservabilityLayer["Observability & Analytics"]
        LangfuseTracer["Langfuse Python Client\n(Tracing Context Managers)"]
        LangfuseServer["Langfuse v2 Web Dashboard\n(Port 3000)"]
        ClickHouse[("ClickHouse Analytics\nPort 8123 / 9000")]
        LangfusePG[("Langfuse Postgres\nPort 5432")]
    end

    subgraph OrchestrationLayer["Data Pipeline & Extraction"]
        Airflow["Apache Airflow 2.8+\nOrchestration DAGs\nPort 8080"]
        Parsers["PDF Parsing Engine\n(Docling / DeepSeek OCR)"]
    end

    %% Client flows
    Gradio --> API
    Telegram --> API
    RestClients --> API

    %% Internal routing
    API --> RouterAsk
    API --> RouterAgentic
    API --> RouterStream
    API --> RouterHealth

    %% Cache & Storage
    RouterAsk <--> Redis
    RouterStream <--> Redis
    RouterAsk --> OpenSearch
    RouterAgentic --> OpenSearch
    API --> Postgres

    %% Inference
    RouterAsk --> JinaAPI
    RouterAgentic --> JinaAPI
    RouterAsk --> Ollama
    RouterAgentic --> Ollama

    %% Observability
    API --> LangfuseTracer
    LangfuseTracer --> LangfuseServer
    LangfuseServer --> ClickHouse
    LangfuseServer --> LangfusePG

    %% Ingestion
    Airflow --> Parsers
    Parsers --> OpenSearch
    Parsers --> Postgres
```

---

## 2. Core Service Topology

The complete infrastructure is orchestrated through Docker Compose:

| Service | Container Name | Image / Base | Internal Port | Host Port | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **API** | `rag-api` | Python 3.12 (Custom) | 8000 | 8000 | FastAPI REST service, routing, business logic |
| **Redis** | `rag-redis` | `redis:7-alpine` | 6379 | 6379 | Sub-second exact-match response caching |
| **OpenSearch** | `rag-opensearch` | `opensearchproject/opensearch:2.19.0` | 9200, 9600 | 9200, 9600 | Vector database + BM25 keyword search index |
| **OpenSearch Dashboards** | `rag-dashboards` | `opensearchproject/opensearch-dashboards:2.19.0` | 5601 | 5601 | Search index management and visualization |
| **Ollama** | `rag-ollama` | `ollama/ollama:0.11.2` | 11434 | 11434 | Local inference engine (Llama-3.2, Qwen, etc.) |
| **PostgreSQL** | `rag-postgres` | `postgres:16-alpine` | 5432 | 5432 | arXiv paper metadata and ingestion state |
| **Langfuse** | `rag-langfuse` | `langfuse/langfuse:2` | 3000 | 3000 | LLM observability, trace visualizer, span monitor |
| **Langfuse Postgres** | `rag-langfuse-postgres` | `postgres:16-alpine` | 5432 | Internal | Dedicated relational store for Langfuse state |
| **ClickHouse** | `rag-clickhouse` | `clickhouse/clickhouse-server:24.8-alpine` | 8123, 9000 | Internal | High-performance analytics backend for Langfuse |
| **Airflow** | `rag-airflow` | Apache Airflow (Custom) | 8080 | 8080 | Scheduled metadata ingestion and PDF indexing |

---

## 3. Query Pipelines & Data Flows

### 3.1 Standard RAG Query Flow (`/api/v1/ask`)

The standard RAG pipeline prioritizes rapid response times by first querying the exact-match cache and falling back to OpenSearch hybrid search:

```mermaid
sequenceDiagram
    autonumber
    actor Client as User / Client
    participant API as FastAPI Router
    participant Cache as Redis Cache
    participant Tracer as Langfuse Tracer
    participant Embed as Jina AI API
    participant Search as OpenSearch Hybrid
    participant LLM as Ollama Inference

    Client->>API: POST /api/v1/ask {query, top_k, use_hybrid, model}
    API->>Tracer: Start Trace (trace_request)
    API->>Cache: Check exact_cache:{sha256_key}
    alt Cache Hit
        Cache-->>API: Return Cached AskResponse
        API->>Tracer: Complete Trace (status=cached, duration < 5ms)
        API-->>Client: 200 OK (Instant Response)
    else Cache Miss
        Cache-->>API: Nil (Cache Miss)
        opt use_hybrid is True
            API->>Tracer: Start Span (query_embedding)
            API->>Embed: Embed query string (1024-dim)
            Embed-->>API: Return dense vector float array
            API->>Tracer: End Span (query_embedding)
        end
        API->>Tracer: Start Span (search_retrieval)
        API->>Search: Execute Hybrid Query (BM25 + k-NN + RRF)
        Search-->>API: Return Top-K Ranked Chunks
        API->>Tracer: End Span (search_retrieval, chunks_returned, unique_papers)
        API->>Tracer: Start Span (prompt_construction)
        API->>API: Build Context-Injected RAG Prompt
        API->>Tracer: End Span (prompt_construction)
        API->>Tracer: Start Span (llm_generation)
        API->>LLM: Generate Answer with Ollama
        LLM-->>API: Synthesized Response
        API->>Tracer: End Span (llm_generation)
        API->>Cache: SET exact_cache:{sha256_key} with TTL (6 hours)
        API->>Tracer: Complete Trace
        API-->>Client: 200 OK (AskResponse with Answer & Paper Sources)
    end
```

### 3.2 Agentic RAG Flow (`/api/v1/agentic_ask`)

The agentic pipeline enforces strict quality gates, domain validation, and adaptive query reformulation:

```mermaid
flowchart TD
    Start([User Question Received]) --> StepGuardrail{Domain Guardrail Classifier}
    
    StepGuardrail -- Score < 60\n(Out of Domain) --> Reject[Return Domain Scope Error\nNo OpenSearch Retrieval\nImmediate Response]
    
    StepGuardrail -- Score >= 60\n(CS / AI Domain) --> StepRewrite{Query Clarity Check}
    
    StepRewrite -- Vague / Short Query --> RewriteQuery[Adaptive Query Rewriter\nExpand with Domain Concepts]
    StepRewrite -- Clear & Technical Query --> SearchDocs[Hybrid Search Retriever]
    RewriteQuery --> SearchDocs
    
    SearchDocs --> GradeDocs{Candidate Chunks Found?}
    
    GradeDocs -- Chunks Available --> PromptConstruct[Structured Grounded Prompting\nInject Citations & Paper IDs]
    GradeDocs -- Zero Chunks Found --> DirectGen[Direct Model Synthesis\nFallback Answer]
    
    PromptConstruct --> OllamaGen[Ollama LLM Generation]
    DirectGen --> OllamaGen
    
    OllamaGen --> AssembleResponse[Assemble AgenticAskResponse\nReasoning Steps + Sources + Attempts]
    AssembleResponse --> End([Deliver Response to Client])
```

---

## 4. Hybrid Search Architecture

The search service utilizes an OpenSearch single hybrid index (`arxiv-papers-chunks`) configured with dense vector k-NN and sparse BM25 inverted indexes.

### 4.1 Hybrid Search Pipeline Mechanics

1. **Sparse Lexical Search (BM25)**:
   - Matches exact keywords, technical terms, symbols, and author metadata.
   - Evaluates term frequency and inverse document frequency over fields `chunk_text`, `title`, and `abstract`.

2. **Dense Vector Search (k-NN)**:
   - Encodes user query into a 1024-dimensional normalized vector via Jina AI (`jina-embeddings-v2-base-en`).
   - Searches `chunk_vector` using cosine similarity (`cosinesimil` space type) with HNSW (Hierarchical Navigable Small World) index graphs.

3. **Reciprocal Rank Fusion (RRF)**:
   - Combines ranked lists from the vector query and BM25 query into a unified relevance score:
     $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
     where $k = 60$, and $r_m(d)$ is the document rank within model $m$.
   - Eliminates score scale differences between cosine similarity $[0, 1]$ and unbounded BM25 scores.

```mermaid
flowchart LR
    Query["User Query"] --> Lexical["OpenSearch BM25\nMatch Query"]
    Query --> Jina["Jina AI Embeddings\n(1024-dim vector)"]
    Jina --> Vector["OpenSearch k-NN\nCosine Similarity"]
    
    Lexical --> RRF["OpenSearch Search Pipeline\nReciprocal Rank Fusion (RRF)"]
    Vector --> RRF
    
    RRF --> RankedResults["Top-K Candidate Chunks\n(arxiv_id, chunk_text, score)"]
```

---

## 5. Exact-Match Redis Caching Architecture

To achieve production-grade throughput and eliminate redundant LLM inference costs, the microservice implements an exact-match caching layer backed by Redis 7.

### 5.1 Cache Key Generation

Cache keys are constructed deterministically from all parameters that affect generation output:

```python
key_data = {
    "query": request.query.strip(),
    "model": request.model,
    "top_k": request.top_k,
    "use_hybrid": request.use_hybrid,
    "categories": sorted(request.categories) if request.categories else [],
}
key_string = json.dumps(key_data, sort_keys=True)
key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
cache_key = f"exact_cache:{key_hash}"
```

### 5.2 Performance Profile & Invalidation

- **Cache Hit Latency**: **1 to 5 milliseconds** ($>100\times$ speedup compared to $12\text{--}25$ seconds cold LLM generation).
- **Time-to-Live (TTL)**: 6 hours (`ttl_hours = 6` default, configurable via `REDIS__TTL_HOURS`).
- **Memory Management**: Redis configured with `maxmemory 256mb` and `maxmemory-policy allkeys-lru` (Least Recently Used eviction policy prevents out-of-memory under high load).
- **Fault-Tolerant Bypass**: If Redis is unreachable or returns a serialization error, the request automatically falls back to full pipeline execution without failing the user request.

---

## 6. Langfuse Observability & Tracing Architecture

Every interaction with the microservice is traced through self-hosted Langfuse v2, providing end-to-end visibility into latencies, token consumption, and retrieval quality.

```mermaid
classDiagram
    class RAGRequestTrace {
        +String trace_id
        +String name: "api_user"
        +String query
        +Float total_duration
        +Dict metadata
    }

    class EmbeddingSpan {
        +String name: "query_embedding"
        +String query
        +Int query_length
        +Float embedding_duration_ms
        +Boolean success
    }

    class SearchSpan {
        +String name: "search_retrieval"
        +String query
        +Int top_k
        +Int chunks_returned
        +Int unique_papers
        +Int total_hits
        +List arxiv_ids
    }

    class PromptSpan {
        +String name: "prompt_construction"
        +Int chunk_count
        +Int prompt_length
        +String prompt_preview
    }

    class GenerationSpan {
        +String name: "llm_generation"
        +String model
        +Int prompt_length
        +String prompt
        +String output_answer
    }

    RAGRequestTrace *-- EmbeddingSpan
    RAGRequestTrace *-- SearchSpan
    RAGRequestTrace *-- PromptSpan
    RAGRequestTrace *-- GenerationSpan
```

### Trace Span Hierarchy
1. **`trace_request`**: Root trace recording user identifier, query text, total execution latency, and success/failure state.
2. **`query_embedding`**: Captures embedding generation time and vector dimensions.
3. **`search_retrieval`**: Records OpenSearch execution time, total documents matched, chunks returned, and paper IDs.
4. **`prompt_construction`**: Records token/character length of the synthesized prompt and chunk count.
5. **`llm_generation`**: Captures the exact model version used, full prompt sent, and final generation tokens.

