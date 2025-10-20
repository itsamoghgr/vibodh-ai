# Vibodh AI

**AI Brain for Your Company** - Intelligent knowledge management and AI-powered insights platform.

---

## Features

### 🧠 **Intelligent Knowledge Management**
- **Semantic Search**: Advanced RAG (Retrieval-Augmented Generation) with OpenAI embeddings (text-embedding-3-small, 1536 dimensions)
- **Hybrid Retrieval**: Combines vector search, knowledge graph context, AI insights, and conversation memory
- **Smart Chunking**: Recursive text splitting with configurable chunk size and overlap
- **In-Memory Caching**: 15-minute TTL for embeddings to optimize performance

### 📊 **Knowledge Graph**
- **Entity Extraction**: Automatically identifies 5 entity types (person, project, topic, tool, issue)
- **Relationship Mapping**: Extracts 10 relationship types (works_on, discussed, fixed, blocked_by, uses, created, mentioned, assigned_to, commented_on)
- **LLM-Powered**: Uses Groq Llama 3.3 70B or OpenAI GPT-4 for extraction
- **Deduplication**: Intelligent entity and relationship deduplication with confidence scoring
- **Graph Visualization**: Full knowledge graph with entities and edges

### 💡 **AI Insights**
- **Organizational Analytics**: Generates insights about projects, teams, trends, and risks
- **Activity Analysis**: Analyzes last 7-30 days of organizational activity
- **Category-based Insights**: project, team, trend, risk, general
- **Confidence Scoring**: Each insight includes confidence level (0.0-1.0)
- **Actionable Recommendations**: Provides specific actions based on patterns

### 💬 **Conversational AI**
- **Streaming Responses**: Real-time streaming chat powered by Groq Llama 3.3 70B Versatile
- **Session Management**: Maintains conversation context across sessions
- **Memory System**: Stores important conversation contexts with importance scoring
- **Personalization**: User-specific responses using AI memory
- **Feedback Loop**: User feedback (positive/negative) for continuous improvement

### 🔄 **Data Ingestion**

#### Slack Integration
- **OAuth 2.0 Flow**: Secure workspace connection
- **Channel Sync**: Public channels (always), private channels (optional with scope)
- **Thread Support**: Ingests message threads and replies
- **File Attachments**: Extracts content from text files
- **User Mentions**: Resolves @mentions to user names
- **Real-time Webhooks**: Live message ingestion via webhook events
- **Deduplication**: Source ID-based duplicate prevention
- **Progress Tracking**: Batch job status monitoring

#### ClickUp Integration
- **OAuth 2.0 Flow**: Secure workspace connection
- **Hierarchical Data**: Workspaces → Spaces → Lists → Tasks
- **Task Metadata**: Status, priority, assignees, due dates, tags
- **Comments**: Fetches and ingests task comments
- **Real-time Webhooks**: Live task updates (created, updated, deleted, commented)
- **Document Normalization**: Standardized format across sources

### 🎯 **Orchestrator (Cognitive Core)**
- **Intent Classification**: Automatically routes queries (question, task, summary, insight, risk)
- **Multi-Module Routing**: Intelligently selects RAG, KG, Insights modules based on intent
- **Meta-Rule Application**: Applies discovered patterns to refine module selection (Phase 3, Step 4)
- **Context Aggregation**: Combines results from multiple modules
- **Reasoning Chain**: Logs decision steps for transparency
- **Parallel Execution**: Runs module queries concurrently for speed
- **Synthesis**: LLM-powered final response generation

### 🧬 **Meta-Learning & Knowledge Evolution** (Phase 3, Step 4)
- **Reasoning Pattern Analysis**: Discovers which module combinations work best for each intent type
- **Meta-Rule Generation**: Creates rules like "For risk queries → KG + Insights (85% success)"
- **Dynamic Orchestration**: Applies meta-rules to refine module routing at runtime
- **KG Schema Evolution**: Proposes new entity/relation types based on data patterns
- **Model Snapshots**: Tracks configuration states before/after optimization cycles
- **Trend Detection**: LLM-powered analysis of emerging topics and recurring concepts
- **Nightly Analysis**: Automatic pattern discovery running at 2 AM UTC
- **Self-Evolution**: AI that learns how to learn and continuously improves

### 🔄 **Adaptive Reasoning** (Phase 3, Step 3)
- **Performance Tracking**: Monitors accuracy, response time, user feedback
- **Automatic Optimization**: Adjusts module weights and LLM parameters
- **Self-Reflection**: Nightly analysis of "what went right/wrong"
- **A/B Testing**: Compares configuration changes
- **Feedback Loop**: Continuous improvement based on results

### 💾 **Long-Term Memory** (Phase 3, Step 2)
- **Memory Consolidation**: Nightly aggregation with importance decay
- **Access Tracking**: Boosts frequently used memories
- **Embedding-Based Retrieval**: Semantic memory search
- **Expiration Management**: Auto-expires low-importance memories

### 🏗️ **Architecture**

#### Clean Modular Design
- **API Layer**: Versioned REST endpoints (`/api/v1/*`) with FastAPI
- **Service Layer**: Business logic isolated from HTTP concerns
- **Connector Layer**: Integration abstractions for external services
- **Data Layer**: Supabase PostgreSQL with vector storage
- **Core Layer**: Configuration, logging, and shared utilities

#### Design Patterns
- Singleton pattern for services
- Factory pattern for service creation
- Dependency injection
- Strategy pattern for LLM providers
- Repository pattern for database access

### 🔒 **Security & Configuration**
- **Row Level Security**: Supabase RLS for data isolation
- **Service Role Keys**: Admin operations with SERVICE_ROLE_KEY
- **Environment Variables**: Type-safe configuration with Pydantic
- **CORS**: Configurable cross-origin resource sharing
- **Structured Logging**: JSON-formatted logs with context

### 📈 **Performance Optimizations**
- **Embeddings Cache**: In-memory with 15-minute TTL
- **Batch Processing**: Efficient batch embedding generation
- **Token Counting**: TikToken for API limit management
- **Lazy Loading**: Services initialized on-demand
- **Connection Pooling**: Supabase client singleton pattern

### 🚀 **Developer Experience**
- **API Versioning**: Future-proof with `/v1/` prefix
- **Auto-generated Docs**: Swagger UI and ReDoc (debug mode)
- **Type Safety**: Full Python type hints with Pydantic models
- **Testing Ready**: Pytest configured with markers
- **Docker Support**: Multi-stage builds with health checks
- **Hot Reload**: Uvicorn development server with auto-reload

---

## API Endpoints

### Core
- `GET /` - Root endpoint
- `GET /health` - Health check

### Chat (`/api/v1/chat`)
- `POST /stream` - Streaming chat with RAG
- `GET /history` - Get chat session history
- `GET /{session_id}` - Get specific session with messages
- `POST /feedback` - Submit message feedback

### Slack (`/api/v1/slack`)
- `GET /connect` - Start OAuth flow
- `GET /connect/callback` - OAuth callback
- `POST /ingest` - Ingest Slack messages
- `POST /events` - Webhook for real-time updates

### ClickUp (`/api/v1/clickup`)
- `GET /connect` - Start OAuth flow
- `GET /callback` - OAuth callback
- `POST /sync/{connection_id}` - Sync tasks and comments
- `POST /webhook` - Webhook for real-time updates

### Knowledge Graph (`/api/v1/kg`)
- `GET /stats/{org_id}` - Get KG statistics
- `GET /entities/{org_id}` - List entities (with type filter)
- `GET /edges/{org_id}` - List relationships/edges

### Insights (`/api/v1/insights`)
- `POST /run/{org_id}` - Generate new insights
- `GET /list/{org_id}` - List existing insights
- `GET /stats/{org_id}` - Get insight statistics

### Connections (`/api/v1/connections`)
- `GET /` - List all connections
- `DELETE /{connection_id}` - Delete a connection

### Documents (`/api/v1/documents`)
- `GET /` - List documents
- `POST /retry-failed-embeddings/{org_id}` - Retry failed embeddings

### Orchestrator (`/api/v1/orchestrate`)
- `POST /query` - Route query through cognitive core
- `GET /logs` - Get reasoning logs
- `GET /logs/{log_id}` - Get detailed reasoning log

### RAG (`/api/v1/rag`)
- `POST /search` - Semantic search with context
- `POST /generate` - Generate answer using RAG

### Memory (`/api/v1/memory`)
- `POST /store` - Store memory
- `GET /retrieve` - Retrieve memories by query
- `POST /consolidate` - Trigger memory consolidation

### Adaptive (`/api/v1/adaptive`)
- `GET /config` - Get adaptive configuration
- `POST /optimize` - Run optimization cycle
- `GET /performance` - Get performance metrics
- `GET /history` - Get optimization history

### Meta-Learning (`/api/v1/meta-learning`) - Phase 3, Step 4
- `POST /analyze` - Trigger meta-learning analysis
- `GET /rules` - Get discovered meta-rules
- `GET /patterns` - Get reasoning pattern statistics
- `GET /kg-evolution` - Get KG schema evolution history
- `POST /kg-evolution/propose` - Propose schema changes
- `POST /kg-evolution/approve` - Approve schema change
- `POST /kg-evolution/reject` - Reject schema change
- `GET /snapshots` - Get model configuration snapshots
- `POST /snapshot/create` - Create new snapshot
- `GET /trends` - Get detected data trends
- `GET /schema-version` - Get current KG schema version

---

## Tech Stack

**Backend Framework**: FastAPI 0.115.12, Uvicorn 0.34.0

**Database**: Supabase 2.15.2 (PostgreSQL with vector storage)

**AI/ML**:
- OpenAI 1.59.8 (text-embedding-3-small)
- Groq 0.32.0 (Llama 3.3 70B Versatile)
- LangChain Text Splitters 0.3.4
- NumPy 1.26.4 (vector operations)

**Integrations**:
- Slack SDK 3.33.5
- HTTPX 0.28.1

**Configuration**:
- Pydantic 2.11.3
- Pydantic Settings 2.7.1
- Python-dotenv 1.0.1

**Testing**:
- Pytest 8.0.0
- Pytest-asyncio 0.23.0

**Deployment**: Docker with multi-stage builds

---

## Project Structure

```
vibodh-ai/
├── app/                        # Main application
│   ├── main.py                 # FastAPI entry point
│   ├── core/                   # Configuration & logging
│   │   ├── config.py           # Pydantic settings
│   │   └── logging.py          # Structured JSON logging
│   ├── db/                     # Database layer
│   │   └── supabase_client.py  # Client factory
│   ├── models/                 # Pydantic models
│   │   ├── document.py
│   │   ├── query.py
│   │   ├── integration.py
│   │   └── legacy_schemas.py
│   ├── services/               # Business logic
│   │   ├── rag_service.py
│   │   ├── kg_service.py
│   │   ├── insight_service.py
│   │   ├── orchestrator_service.py
│   │   ├── embedding_service.py
│   │   ├── llm_service.py
│   │   ├── ingestion_service.py
│   │   ├── clickup_service.py
│   │   ├── memory_service.py
│   │   ├── adaptive_engine.py
│   │   ├── feedback_service.py
│   │   └── meta_learning_service.py  # NEW!
│   ├── connectors/             # Integration abstractions
│   │   ├── slack_connector.py
│   │   └── clickup_connector.py
│   ├── api/v1/                 # Versioned API routes
│   │   ├── __init__.py         # Router aggregator
│   │   ├── routes_chat.py
│   │   ├── routes_slack.py
│   │   ├── routes_clickup.py
│   │   ├── routes_kg.py
│   │   ├── routes_insights.py
│   │   ├── routes_connections.py
│   │   ├── routes_documents.py
│   │   ├── routes_orchestrator.py
│   │   ├── routes_rag.py
│   │   ├── routes_memory.py
│   │   ├── routes_adaptive.py
│   │   └── routes_meta_learning.py  # NEW!
│   ├── utils/                  # Utilities
│   └── tests/                  # Test suite
│       ├── unit/
│       └── integration/
├── connectors/                 # Legacy OAuth connectors
│   └── slack_connector.py
├── .env                        # Environment variables
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── pytest.ini
```

---

## License

Proprietary - All rights reserved
