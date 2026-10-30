# Operations & Benchmarking Guide

This guide outlines setup procedures, container orchestration, environment variable configurations, Redis cache monitoring, Langfuse observability workflows, and performance benchmarking for the Agentic RAG microservice.

---

## 1. System Requirements & Prerequisites

- **Operating System**: Linux (Ubuntu 22.04+ recommended), macOS (Apple Silicon supported), or Windows 11 with WSL2.
- **Docker Engine**: Version 24.0+ with Docker Compose v2.20+.
- **Memory**: Minimum 16 GB RAM (32 GB recommended if running larger Ollama models locally).
- **Disk Space**: At least 30 GB free for Docker images, OpenSearch indices, and model weights.
- **Python**: 3.12 (if running local development scripts or notebooks outside Docker).
- **Jina AI API Key**: Required for dense vector embeddings (`https://jina.ai/embeddings/`).

---

## 2. Step-by-Step Developer Onboarding

### Step 1: Clone Repository & Create Environment Configuration
```bash
git clone https://github.com/your-org/agentic-rag-research-assistant.git
cd agentic-rag-research-assistant

# Create local environment file from template
cp .env.example .env
```

Edit `.env` to supply your required credentials:
```ini
# Add your Jina AI key for hybrid vector embeddings
JINA_API_KEY=jina_xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Self-hosted Langfuse keys (generated in Step 4)
LANGFUSE__PUBLIC_KEY=pk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
LANGFUSE__SECRET_KEY=sk-lf-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### Step 2: Launch Complete Microservice Stack with Docker Compose
```bash
docker compose up --build -d
```

Verify that all services reach a healthy status:
```bash
docker compose ps
```

Expected container states:
| Service | Container | Status |
| :--- | :--- | :--- |
| `api` | `rag-api` | `Up (healthy)` on `0.0.0.0:8000` |
| `redis` | `rag-redis` | `Up (healthy)` on `0.0.0.0:6379` |
| `opensearch` | `rag-opensearch` | `Up (healthy)` on `0.0.0.0:9200, 9600` |
| `opensearch-dashboards`| `rag-dashboards` | `Up (healthy)` on `0.0.0.0:5601` |
| `ollama` | `rag-ollama` | `Up (healthy)` on `0.0.0.0:11434` |
| `postgres` | `rag-postgres` | `Up (healthy)` on `0.0.0.0:5432` |
| `langfuse` | `rag-langfuse` | `Up (healthy)` on `0.0.0.0:3000` |
| `langfuse-postgres` | `rag-langfuse-postgres` | `Up (healthy)` |
| `clickhouse` | `rag-clickhouse` | `Up (healthy)` |
| `airflow` | `rag-airflow` | `Up (healthy)` on `0.0.0.0:8080` |

### Step 3: Pull LLM Models in Ollama
Once the `rag-ollama` container is healthy, pull the default production model:
```bash
docker exec -it rag-ollama ollama pull llama3.2:1b
```

*(Optional) Pull higher-capacity models if GPU memory permits:*
```bash
docker exec -it rag-ollama ollama pull llama3.2:3b
```

### Step 4: Configure Self-Hosted Langfuse Observability
1. Open your browser and navigate to `http://localhost:3000`.
2. Complete the initial admin sign-up (stored in your dedicated `rag-langfuse-postgres` database).
3. Create a new project (e.g. `arxiv-agentic-rag`).
4. Navigate to **Settings $\rightarrow$ API Keys** and click **Create new API keys**.
5. Copy the `Public Key` and `Secret Key` and update your `.env` file:
   ```ini
   LANGFUSE__ENABLED=true
   LANGFUSE__HOST=http://localhost:3000
   LANGFUSE__PUBLIC_KEY=pk-lf-...
   LANGFUSE__SECRET_KEY=sk-lf-...
   ```
6. Restart the API container to load the active credentials:
   ```bash
   docker compose restart api
   ```

### Step 5: Verify Readiness via Health Endpoint
```bash
curl -s http://localhost:8000/api/v1/health | jq .
```

---

## 3. Environment Variables Reference

The microservice uses nested Pydantic settings with double-underscore delimiters:

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `ENVIRONMENT` | string | `"development"` | Active runtime mode (`development`, `staging`, `production`) |
| `POSTGRES_DATABASE_URL` | string | `postgresql+psycopg2://rag_user:rag_password@postgres:5432/rag_db` | Relational database connection string |
| `OPENSEARCH_HOST` | string | `http://opensearch:9200` | OpenSearch endpoint URL |
| `OPENSEARCH__INDEX_NAME` | string | `"arxiv-papers"` | Base index prefix |
| `OPENSEARCH__CHUNK_INDEX_SUFFIX` | string | `"chunks"` | Suffix creating `arxiv-papers-chunks` |
| `OPENSEARCH__VECTOR_DIMENSION` | int | `1024` | Jina embedding dimension |
| `OPENSEARCH__VECTOR_SPACE_TYPE` | string | `"cosinesimil"` | Vector metric (`cosinesimil`, `l2`, `innerproduct`) |
| `OLLAMA_HOST` | string | `http://ollama:11434` | Ollama inference API URL |
| `OLLAMA_MODEL` | string | `"llama3.2:1b"` | Default LLM identifier |
| `REDIS__HOST` | string | `"localhost"` (or `redis`) | Redis host address |
| `REDIS__PORT` | int | `6379` | Redis TCP port |
| `REDIS__TTL_HOURS` | int | `6` | Exact-match cache time-to-live |
| `JINA_API_KEY` | string | `""` | Jina AI vector cloud API key |
| `LANGFUSE__ENABLED` | bool | `true` | Toggle for tracing middleware |
| `LANGFUSE__HOST` | string | `http://localhost:3000` | Langfuse web server URL |
| `LANGFUSE__PUBLIC_KEY` | string | `""` | Langfuse project public API key |
| `LANGFUSE__SECRET_KEY` | string | `""` | Langfuse project secret API key |

---

## 4. Redis Cache Monitoring & Administration

### 4.1 Redis CLI Access
Connect to the running Redis container:
```bash
docker exec -it rag-redis redis-cli
```

### 4.2 Inspecting Cache Performance & Keys
```redis
# Inspect server info and hit/miss counts
INFO stats

# List currently cached query keys
KEYS exact_cache:*

# View TTL remaining on a specific cached query (in seconds)
TTL exact_cache:<hash>

# Inspect cached payload
GET exact_cache:<hash>

# Live stream incoming cache operations
MONITOR
```

### 4.3 Cache Memory Management
Redis is configured in `compose.yml` with:
- `--maxmemory 256mb`
- `--maxmemory-policy allkeys-lru`

When cache storage approaches 256MB, Redis automatically evicts the least recently accessed query responses, ensuring zero downtime and zero memory exhaustion.

---

## 5. Langfuse Observability Dashboard Usage

Self-hosted Langfuse provides a comprehensive view into the microservice's inference pipeline:

```mermaid
flowchart TD
    Dashboard["Langfuse Dashboard\nhttp://localhost:3000"]
    
    Dashboard --> TracesView["Traces View\n- Total Request Duration\n- User Session ID\n- Error Rates"]
    Dashboard --> SpansView["Span Detail Breakdown\n- query_embedding (ms)\n- search_retrieval (hits & IDs)\n- prompt_construction\n- llm_generation (tokens/s)"]
    Dashboard --> MetricsView["Model Analytics\n- Token usage distribution\n- Latency p50, p90, p99\n- Fallback frequency"]
```

### Key Tracing Workflows:
1. **Inspecting Latency Bottlenecks**: Open any trace to see a waterfall diagram detailing exactly how many milliseconds were consumed by Jina AI embedding generation vs OpenSearch k-NN search vs Ollama token generation.
2. **Auditing Retrieval Quality**: Open the `search_retrieval` span to inspect the exact `arxiv_ids` returned by OpenSearch for any user query.
3. **Validating Prompts**: Open the `prompt_construction` span to see the exact formatted prompt and chunk insertions sent to Ollama.

---

## 6. Benchmarking & Performance Testing

### 6.1 Exact-Match Cache Latency Benchmark

Testing identical queries against `/api/v1/ask` demonstrates dramatic latency reduction:

| Metric | Cache Miss (Cold Request) | Cache Hit (Warm Request) | Optimization Impact |
| :--- | :--- | :--- | :--- |
| **Response Latency** | $12.40\text{--}18.50\text{ s}$ | $\mathbf{0.003\text{--}0.005\text{ s}}$ ($3\text{--}5\text{ ms}$) | **$>100\times$ Speedup** |
| **OpenSearch Load** | Full hybrid query execution | $0$ queries executed | $100\%$ DB offload |
| **LLM Compute** | Full generation tokens | $0$ tokens consumed | Zero compute cost |
| **User Experience** | Noticeable generation wait | Instantaneous delivery | Seamless UI feedback |

### 6.2 Executing Automated Verification Notebooks

The repository provides automated Jupyter verification notebooks in `notebooks/`:

```bash
# Start Jupyter server
jupyter notebook --ip 0.0.0.0 --port 8888
```

1. **`notebooks/Agentic_RAG_with_LangGraph.ipynb`**:
   - Tests out-of-scope guardrail rejection (verifying score $<60$).
   - Tests adaptive query rewriting on underspecified queries.
   - Tests grounded response generation with arXiv citations.
2. **`notebooks/Production_RAG_with_Caching_Observability.ipynb`**:
   - Verifies all service health endpoints.
   - Executes baseline vs cached benchmark queries and calculates exact speedup factors.
   - Confirms Redis persistence and TTL validation.

### 6.3 Automated Test Suite Execution

Run unit and integration tests using pytest:
```bash
python -m pytest -p no:dotenv tests/unit
```

