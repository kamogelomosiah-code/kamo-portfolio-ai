import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { InferenceClient } from '@huggingface/inference';
import { Pinecone } from '@pinecone-database/pinecone';

const app = express();
const port = process.env.PORT || 3000;

if (!process.env.HF_TOKEN) throw new Error('Missing HF_TOKEN in environment variables.');
if (!process.env.PINECONE_API_KEY) throw new Error('Missing PINECONE_API_KEY in environment variables.');

const hf = new InferenceClient(process.env.HF_TOKEN);
const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.Index('kamo-portfolio');

app.use(cors());
app.use(express.json({ limit: '2mb' }));

const embedText = async (text) => {
  const response = await hf.featureExtraction({
    model: 'sentence-transformers/all-MiniLM-L6-v2',
    inputs: text
  });

  let vector = response;
  while (Array.isArray(vector) && Array.isArray(vector[0])) vector = vector[0];
  if (!Array.isArray(vector)) throw new Error('Embedding output was not a valid array.');
  return vector.map(Number);
};

const getRelevantContext = async (message) => {
  const matches = await index.query({
    vector: await embedText(message),
    topK: 3,
    includeMetadata: true,
    includeValues: false
  });
  return matches.matches || [];
};

const buildSystemPrompt = (matches, additionalContext) => {
  const retrievedContext = matches
    .map((match) => match.metadata?.text || match.metadata?.content || '')
    .filter(Boolean)
    .join('\n\n---\n\n');

  return [
    "You are Kamo's AI, the portfolio assistant for Kamogelo.",
    'Answer questions about Kamogelo using the portfolio context below.',
    'If the answer is not in the context, say so clearly and do not invent facts.',
    'Be helpful, concise, professional, and warm.',
    '',
    'Portfolio Context:',
    retrievedContext || 'No relevant portfolio context was found. Avoid guessing.',
    ...(additionalContext ? ['', 'Additional Context from the user:', additionalContext] : [])
  ].join('\n');
};

app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.post('/api/chat', async (req, res) => {
  try {
    const { message, additionalContext } = req.body || {};
    if (typeof message !== 'string' || !message.trim()) {
      return res.status(400).json({ error: 'A non-empty "message" field is required.' });
    }

    const systemPrompt = buildSystemPrompt(
      await getRelevantContext(message),
      typeof additionalContext === 'string' ? additionalContext.trim() : ''
    );

    const completion = await hf.chatCompletion({
      model: 'Qwen/Qwen2.5-7B-Instruct',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: message.trim() }
      ],
      temperature: 0.3,
      max_tokens: 500
    });

    const reply = completion?.choices?.[0]?.message?.content || completion?.generated_text;
    if (!reply) throw new Error('The model returned an empty response.');
    res.json({ reply: reply.trim() });
  } catch (error) {
    console.error('Chat endpoint error:', error);
    res.status(500).json({ error: 'Failed to process the request.' });
  }
});

app.listen(port, () => console.log(`Server is running on port ${port}`));
