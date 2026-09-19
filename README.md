# Kamo's AI Portfolio Assistant

A Node.js and Express backend for an AI portfolio assistant using Hugging Face Inference Providers, Pinecone, and Retrieval-Augmented Generation (RAG).

## Prerequisites

- Node.js 18 or newer
- A Hugging Face access token
- A Pinecone account and API key

## Local setup

```bash
npm install
cp .env.example .env
```

Set the values in `.env`:

```env
HF_TOKEN=your_huggingface_token
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_HOST=your_pinecone_index_host
PORT=3000
```

Create a Pinecone index named `kamo-portfolio` with:

- Dimension: `384`
- Metric: `cosine`

Replace the placeholder `knowledgeBase` data in `ingest.js` with your CV and portfolio content, then ingest it:

```bash
npm run ingest
```

Start the API:

```bash
npm start
```

## API

Health check:

```bash
curl http://localhost:3000/health
```

Chat request:

```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Tell me about Kamogelo’s AI and backend skills."}'
```

The chat endpoint embeds the user message, retrieves the top three Pinecone matches, injects the results into the system prompt, and generates a grounded response with Qwen.

## Deploying to Render

Create a Render Web Service connected to this repository with:

- Build command: `npm install`
- Start command: `npm start`

Add these environment variables in Render:

- `HF_TOKEN`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_HOST`
- `PORT` (Render may provide this automatically)

Run `npm run ingest` locally or through a one-off Render shell after creating the Pinecone index. Do not commit `.env` or secret tokens.

## Production recommendations

- Replace the placeholder knowledge base with verified CV and project data.
- Add rate limiting, authentication, request logging, and stricter CORS settings before exposing the endpoint publicly.
- Keep the Pinecone index dimension at 384 to match `all-MiniLM-L6-v2`.
