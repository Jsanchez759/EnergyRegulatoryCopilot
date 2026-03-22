# Energy Regulatory Copilot

An AI-powered assistant specialized in Colombian energy regulatory matters. This application combines Retrieval-Augmented Generation (RAG) with large language models to provide accurate, context-aware answers about energy regulations.

## Overview

Energy Regulatory Copilot is a full-stack web application designed to help users understand and navigate Colombian energy regulations. It features:

- **Conversational AI**: Ask natural language questions about energy regulations with chat history support
- **Document Indexing**: Upload and index PDF documents (regulatory documents, guidelines, etc.)
- **Retrieval-Augmented Generation (RAG)**: Query indexed documents with AI-powered semantic search
- **Streaming Responses**: Real-time response streaming for improved user experience
- **Multi-mode Chat**: Switch between general question answering and document-based RAG queries

## Architecture

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Vector Database**: Chroma (persistent vector storage for document embeddings)
- **LLM Integration**: OpenRouter (compatible with OpenAI API)
- **PDF Processing**: PyPDF for document parsing and chunking
- **API Style**: RESTful with streaming support

### Frontend
- **Framework**: React 18 with Vite
- **Styling**: CSS
- **Markdown Support**: React Markdown with GitHub Flavored Markdown
- **State Management**: React Hooks (useState, useEffect, useMemo)

## Prerequisites

- Python 3.11 or higher
- Node.js 18+ and npm
- OpenRouter API key (for LLM access)
- Docker and Docker Compose (optional, for containerized deployment)

## Installation

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the backend root with your configuration:
```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_CHAT_MODEL=openrouter/free
OPENROUTER_EMBEDDING_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2:free
VECTOR_DB_PATH=./storage/chroma
LOG_LEVEL=INFO
```

5. Run the backend server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173` (or another port if 5173 is in use).

## Configuration

### Backend Configuration

All backend settings are managed through environment variables and can be customized in `backend/app/core/config.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_NAME` | Energy Regulatory Copilot API | Application title |
| `VERSION` | 0.1.0 | API version |
| `API_V1_PREFIX` | /api/v1 | API endpoint prefix |
| `OPENROUTER_API_KEY` | (empty) | **Required**: Your OpenRouter API key |
| `OPENROUTER_CHAT_MODEL` | openrouter/free | LLM model for chat responses |
| `OPENROUTER_EMBEDDING_MODEL` | nvidia/llama-nemotron-embed-vl-1b-v2:free | Model for generating document embeddings |
| `VECTOR_DB_PATH` | ./storage/chroma | Path to vector database storage |
| `VECTOR_COLLECTION_NAME` | regulatory_documents | Name of the vector collection |
| `RAG_CHUNK_SIZE` | 900 | Size of document chunks (characters) |
| `RAG_CHUNK_OVERLAP` | 200 | Overlap between consecutive chunks |
| `LOG_LEVEL` | INFO | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `ALLOWED_ORIGINS` | localhost:3000, 127.0.0.1:3000 | CORS allowed origins |

### CORS Configuration

For production deployments, update `ALLOWED_ORIGINS` in the `.env` file to include your frontend domain.

## API Endpoints

### Health Check
- **GET** `/api/v1/health`
  - Returns application status

### Questions (General Q&A)
- **POST** `/api/v1/questions/ask`
  - Answer general questions about energy regulations
  - Request: `{ "question": "string", "history": [] }`
  - Response: `{ "answer": "string", "model": "string" }`

- **POST** `/api/v1/questions/ask/stream`
  - Streaming endpoint for real-time responses
  - Returns Server-Sent Events (SSE) stream

### RAG (Document-Based Q&A)
- **POST** `/api/v1/rag/documents/upload`
  - Upload and index a PDF document
  - Requires: PDF file upload, optional OpenRouter API key header
  - Response: `{ "document_id": "uuid", "chunks_indexed": int, "filename": "string" }`

- **POST** `/api/v1/rag/question`
  - Query indexed documents
  - Request: `{ "question": "string", "api_key_override": "string" }`
  - Response: `{ "answer": "string", "sources": [...], "model": "string" }`

- **POST** `/api/v1/rag/question/stream`
  - Streaming document-based Q&A

- **GET** `/api/v1/rag/documents`
  - List all indexed documents
  - Response: `{ "documents": [IndexedDocument], "total": int }`

## Usage

### General Workflow

1. **Start Services**:
   - Backend: `uvicorn app.main:app --reload`
   - Frontend: `npm run dev`

2. **Access the Application**:
   - Open `http://localhost:5173` in your browser

3. **Chat Interface**:
   - Type questions about energy regulations
   - Switch between "Ask" mode (general Q&A) and "RAG" mode (document-based)

4. **Document Upload** (RAG Mode):
   - Click "Upload Document" to add PDF files
   - Wait for indexing to complete
   - Query the uploaded documents in RAG mode

### Example Questions

- "What are the main Colombian energy regulations?"
- "Explain the renewable energy incentives in Colombia"
- (In RAG mode with uploaded documents): "What does the document say about tariffs?"

## Project Structure

```
EnergyRegulatoryCopilot/
├── backend/
│   ├── app/
│   │   ├── api/v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── health.py      # Health check endpoint
│   │   │   │   ├── questions.py   # General Q&A endpoints
│   │   │   │   └── rag.py         # RAG endpoints
│   │   │   └── router.py          # API router configuration
│   │   ├── services/
│   │   │   ├── question_service.py  # Q&A logic
│   │   │   └── rag_service.py       # RAG/document handling
│   │   ├── schemas/
│   │   │   ├── questions.py       # Request/response models
│   │   │   └── rag.py             # RAG models
│   │   ├── core/
│   │   │   ├── config.py          # Configuration settings
│   │   │   └── logging_config.py  # Logging setup
│   │   └── main.py                # FastAPI app initialization
│   ├── requirements.txt           # Python dependencies
│   └── storage/
│       └── chroma/                # Vector database storage
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # Main React component
│   │   ├── main.jsx               # React entry point
│   │   └── styles.css             # Application styles
│   ├── package.json               # NPM dependencies
│   ├── vite.config.js             # Vite configuration
│   └── index.html                 # HTML template
├── docker-compose.yml             # Docker orchestration
└── README.md                      # This file
```

## Dependencies

### Backend
- **fastapi**: Web framework
- **uvicorn**: ASGI server
- **pydantic**: Data validation
- **chromadb**: Vector database
- **openai**: LLM client (OpenRouter compatible)
- **pypdf**: PDF parsing

### Frontend
- **react**: UI framework
- **react-dom**: DOM rendering
- **react-markdown**: Markdown rendering
- **remark-gfm**: GitHub Flavored Markdown support
- **vite**: Build tool

## Development

### Code Style
The project uses Python and JavaScript conventions. Consider using:
- **Backend**: Black for formatting, flake8/ruff for linting
- **Frontend**: ESLint, Prettier

### Debugging

**Backend**:
- Set `LOG_LEVEL=DEBUG` in `.env`
- Check logs in console output
- Use FastAPI interactive docs: `http://localhost:8000/docs`

**Frontend**:
- Open browser DevTools (F12)
- Check Network tab for API requests
- Use React DevTools browser extension

## Deployment

### Docker

Build and run with Docker Compose:
```bash
docker-compose up --build
```

Update `docker-compose.yml` with appropriate environment variables and port mappings.

### Production Considerations

1. **API Keys**: Use secure environment variable management (AWS Secrets, HashiCorp Vault, etc.)
2. **CORS**: Update `ALLOWED_ORIGINS` for your production domain
3. **Database**: Configure persistent storage for Chroma
4. **Rate Limiting**: Implement API rate limiting to prevent abuse
5. **Monitoring**: Set up logging and error tracking (Sentry, CloudWatch, etc.)
6. **HTTPS**: Ensure all communication uses HTTPS

## Troubleshooting

### Common Issues

**Backend won't start**:
- Ensure `OPENROUTER_API_KEY` is set
- Check Python version: `python --version` (should be 3.11+)
- Verify port 8000 is available

**Frontend can't reach backend**:
- Ensure backend is running on `http://localhost:8000`
- Check CORS configuration in backend
- Verify API key header format in requests

**Document upload fails**:
- Ensure PDF is not corrupted
- Check file size isn't excessive
- Verify OpenRouter API key is valid

**Responses are slow**:
- Check OpenRouter API status
- Monitor system resources (CPU, memory)
- Consider adjusting chunk size for RAG

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For issues, questions, or suggestions:
- Create an issue in the repository
- Check existing documentation
- Review API documentation at `http://localhost:8000/docs`

## Roadmap

- [ ] User authentication and authorization
- [ ] Document management (delete, update)
- [ ] Advanced search filters
- [ ] Export chat history
- [ ] Multiple language support
- [ ] Performance optimizations
- [ ] Comprehensive test suite
