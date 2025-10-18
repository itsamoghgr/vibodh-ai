# Vibodh AI - Backend API

FastAPI-powered backend service for Vibodh AI, an intelligent knowledge management platform that integrates with Slack, ClickUp, and other productivity tools to provide AI-powered insights across your company's data.

## Features

- 🤖 **AI-Powered Chat**: Context-aware conversations using Groq (Llama 3.3 70B)
- 🔍 **Semantic Search**: Vector embeddings with OpenAI's text-embedding-3-small
- 🧠 **Knowledge Graph**: Automatic entity and relationship extraction
- 💬 **Slack Integration**: Real-time message sync with webhook support
- ✅ **ClickUp Integration**: Task and project data synchronization
- 🔐 **OAuth 2.0**: Secure authentication for all integrations
- 📊 **Document Ingestion**: Automated data processing and embedding generation

## Tech Stack

- **Framework**: FastAPI
- **Database**: Supabase (PostgreSQL)
- **AI/ML**:
  - OpenAI (embeddings)
  - Groq (LLM completions)
- **Authentication**: Supabase Auth + OAuth 2.0
- **Real-time**: Webhooks for Slack and ClickUp

## Prerequisites

- Python 3.8+
- Supabase account
- OpenAI API key
- Groq API key
- Slack app credentials (optional)
- ClickUp app credentials (optional)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd vibodh-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` with your credentials:
   ```env
   # Supabase
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

   # AI APIs
   OPENAI_API_KEY=your_openai_key
   GROQ_API_KEY=your_groq_key

   # Slack OAuth (optional)
   SLACK_CLIENT_ID=your_slack_client_id
   SLACK_CLIENT_SECRET=your_slack_client_secret
   SLACK_REDIRECT_URI=http://localhost:8000/api/connect/slack/callback

   # ClickUp OAuth (optional)
   CLICKUP_CLIENT_ID=your_clickup_client_id
   CLICKUP_CLIENT_SECRET=your_clickup_client_secret
   CLICKUP_REDIRECT_URI=http://localhost:8000/api/clickup/callback
   CLICKUP_WEBHOOK_URL=http://localhost:8000/api/clickup/webhook

   # Backend URL
   BACKEND_URL=http://localhost:8000
   ```

5. **Set up Supabase database**

   Run the SQL migrations in order:
   ```bash
   # In Supabase SQL editor, run these files in order:
   sql/01_initial_schema.sql
   sql/02_add_channel_fields.sql
   sql/03_add_embeddings_table.sql
   sql/04_add_ingestion_jobs.sql
   sql/05_add_knowledge_graph.sql
   sql/06_add_events_table.sql
   sql/07_add_clickup_source.sql
   ```

## Running the Server

### Development
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, access the interactive API docs:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Key Endpoints

### Authentication
- `POST /auth/signup` - Create new user account
- `POST /auth/login` - User login

### Chat
- `POST /chat` - Send message and get AI response
- `GET /chat/history` - Retrieve chat history

### Integrations
- `GET /api/connect/slack/authorize` - Start Slack OAuth flow
- `GET /api/connect/slack/callback` - Slack OAuth callback
- `POST /api/slack/sync/{connection_id}` - Manual Slack sync
- `POST /api/slack/events` - Slack webhook endpoint

- `GET /api/clickup/authorize` - Start ClickUp OAuth flow
- `GET /api/clickup/callback` - ClickUp OAuth callback
- `POST /api/clickup/sync/{connection_id}` - Manual ClickUp sync
- `POST /api/clickup/webhook` - ClickUp webhook endpoint

### Documents & Search
- `GET /documents` - List all documents
- `POST /search` - Semantic search across documents

### Knowledge Graph
- `GET /knowledge-graph` - Get organization's knowledge graph
- `GET /knowledge-graph/entities` - List entities
- `GET /knowledge-graph/relationships` - List relationships

## Project Structure

```
vibodh-ai/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
├── services/
│   ├── clickup_service.py  # ClickUp integration logic
│   ├── ingestion_service.py # Document ingestion pipeline
│   ├── embedding_service.py # Vector embedding generation
│   ├── kg_service.py       # Knowledge graph extraction
│   └── rag_service.py      # Retrieval-augmented generation
├── connectors/
│   └── slack_connector.py  # Slack API wrapper
└── sql/
    ├── 01_initial_schema.sql
    ├── 02_add_channel_fields.sql
    ├── 03_add_embeddings_table.sql
    ├── 04_add_ingestion_jobs.sql
    ├── 05_add_knowledge_graph.sql
    ├── 06_add_events_table.sql
    └── 07_add_clickup_source.sql
```

## Setting Up Integrations

### Slack Integration

1. **Create Slack App**
   - Go to https://api.slack.com/apps
   - Click "Create New App" → "From scratch"
   - Add OAuth scopes:
     - `channels:history`
     - `channels:read`
     - `groups:history`
     - `groups:read`
     - `users:read`
     - `chat:write`
   - Set redirect URL: `http://localhost:8000/api/connect/slack/callback`
   - Enable Event Subscriptions
   - Add webhook URL: `http://localhost:8000/api/slack/events`
   - Subscribe to bot events: `message.channels`, `message.groups`

2. **Update .env**
   - Copy Client ID and Client Secret to `.env`

### ClickUp Integration

1. **Create ClickUp App**
   - Go to https://app.clickup.com/settings/apps
   - Click "Create an App"
   - Set redirect URL: `http://localhost:8000/api/clickup/callback`

2. **Update .env**
   - Copy Client ID and Client Secret to `.env`

### Public URLs for Webhooks

For webhooks to work in development, use ngrok:

```bash
ngrok http 8000
```

Update `.env` with the ngrok URL:
```env
SLACK_REDIRECT_URI=https://your-ngrok-url.ngrok-free.app/api/connect/slack/callback
CLICKUP_REDIRECT_URI=https://your-ngrok-url.ngrok-free.app/api/clickup/callback
CLICKUP_WEBHOOK_URL=https://your-ngrok-url.ngrok-free.app/api/clickup/webhook
BACKEND_URL=https://your-ngrok-url.ngrok-free.app
```

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `SUPABASE_URL` | Your Supabase project URL | ✅ |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key | ✅ |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | ✅ |
| `GROQ_API_KEY` | Groq API key for LLM | ✅ |
| `PORT` | Server port (default: 8000) | ❌ |
| `SLACK_CLIENT_ID` | Slack OAuth client ID | ⚠️ |
| `SLACK_CLIENT_SECRET` | Slack OAuth client secret | ⚠️ |
| `SLACK_REDIRECT_URI` | Slack OAuth redirect URI | ⚠️ |
| `CLICKUP_CLIENT_ID` | ClickUp OAuth client ID | ⚠️ |
| `CLICKUP_CLIENT_SECRET` | ClickUp OAuth client secret | ⚠️ |
| `CLICKUP_REDIRECT_URI` | ClickUp OAuth redirect URI | ⚠️ |
| `CLICKUP_WEBHOOK_URL` | ClickUp webhook endpoint URL | ⚠️ |
| `BACKEND_URL` | Backend base URL for webhooks | ⚠️ |

⚠️ = Required only if using that integration

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
```

### Linting
```bash
flake8 .
```

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError`
- **Solution**: Make sure virtual environment is activated and dependencies are installed

**Issue**: Slack webhook not receiving events
- **Solution**:
  - Verify ngrok is running
  - Update Slack app webhook URL with ngrok URL
  - Check Event Subscriptions are enabled

**Issue**: ClickUp sync fails with NoneType error
- **Solution**: Some ClickUp tasks may have null values for status/priority/creator. This is handled in the latest version.

**Issue**: Duplicate key error on re-sync
- **Solution**: Old embeddings are now automatically deleted before regeneration.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Your License Here]

## Support

For issues and questions:
- Open an issue on GitHub
- Contact: [Your Contact Info]
