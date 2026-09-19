import 'dotenv/config';
import { InferenceClient } from '@huggingface/inference';
import { Pinecone } from '@pinecone-database/pinecone';

const hfToken = process.env.HF_TOKEN;
const pineconeApiKey = process.env.PINECONE_API_KEY;
const indexName = 'kamogelosportfolio';

if (!hfToken) throw new Error('Missing HF_TOKEN in environment variables.');
if (!pineconeApiKey) throw new Error('Missing PINECONE_API_KEY in environment variables.');

const hf = new InferenceClient(hfToken);
const pinecone = new Pinecone({ apiKey: pineconeApiKey });
const index = pinecone.Index(indexName);

const knowledgeBase = [
  {
    id: 'kamo-about',
    category: 'bio',
    text: 'Kamogelo is a full-stack software developer and AI-focused engineer who enjoys building scalable web products, automation tools, and practical AI experiences.'
  },
  {
    id: 'kamo-skills-frontend',
    category: 'skills',
    text: 'Kamogelo works with JavaScript, TypeScript, React, Next.js, Node.js, Express, and modern frontend tooling to build performant and user-friendly applications.'
  },
  {
    id: 'kamo-skills-backend',
    category: 'skills',
    text: 'Kamogelo builds backend systems with Node.js, RESTful APIs, authentication flows, PostgreSQL, MongoDB, and cloud deployment strategies for production-ready applications.'
  },
  {
    id: 'kamo-ai-experience',
    category: 'experience',
    text: 'Kamogelo is interested in AI engineering and RAG pipelines, integrating embeddings, vector databases, and LLM APIs to create intelligent assistants and portfolio tools.'
  },
  {
    id: 'kamo-portfolio-goal',
    category: 'portfolio',
    text: 'The goal of Kamogelo’s portfolio is to showcase technical capability, problem-solving ability, and the practical application of modern web and AI technologies.'
  },
  {
    id: 'kamo-values',
    category: 'profile',
    text: 'Kamogelo values clean architecture, maintainable code, user-focused product thinking, and continuous learning in software development and AI.'
  }
];

const normalizeEmbedding = (embedding) => {
  let vector = embedding;
  while (Array.isArray(vector) && Array.isArray(vector[0])) vector = vector[0];
  if (!Array.isArray(vector)) throw new Error('Embedding output is not an array.');
  return vector.map(Number);
};

const createEmbedding = async (text) => {
  const response = await hf.featureExtraction({
    model: 'sentence-transformers/all-MiniLM-L6-v2',
    inputs: text
  });
  return normalizeEmbedding(response);
};

const ensureIndexExists = async () => {
  const result = await pinecone.listIndexes();
  const indexes = result?.indexes || [];
  if (!indexes.some((item) => item.name === indexName)) {
    throw new Error(`Pinecone index "${indexName}" does not exist. Create it with dimension 384 and cosine metric.`);
  }
};

const ingest = async () => {
  await ensureIndexExists();
  console.log(`Starting ingestion for ${knowledgeBase.length} chunks...`);

  for (const [position, item] of knowledgeBase.entries()) {
    try {
      const values = await createEmbedding(item.text);
      if (values.length !== 384) throw new Error(`Expected a 384-dimensional vector, received ${values.length}.`);

      await index.upsert({
        vectors: [{
          id: item.id,
          values,
          metadata: { text: item.text, category: item.category }
        }]
      });
      console.log(`✅ Ingested ${position + 1}/${knowledgeBase.length}: ${item.id}`);
    } catch (error) {
      console.error(`❌ Failed to ingest ${item.id}:`, error.message || error);
      throw error;
    }
  }

  console.log('✅ Ingestion complete.');
};

ingest().catch((error) => {
  console.error('❌ Ingestion failed:', error.message || error);
  process.exit(1);
});
