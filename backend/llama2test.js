import path from 'path';
import readline from 'readline';
import { LLM } from 'llama-node';
import { RetrievalQAChain } from 'langchain/chains';
import { LLamaCpp } from 'llama-node/dist/llm/llama-cpp.js';

import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { LLamaEmbeddings } from 'llama-node/dist/extensions/langchain.js';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';

const config = {
  modelPath: path.resolve(process.cwd(), './models/llama-2-7b-chat.ggmlv3.q5_0.bin'),
  enableLogging: true,
  nCtx: 1024,
  seed: 0,
  f16Kv: false,
  logitsAll: false,
  vocabOnly: false,
  useMlock: false,
  embedding: true,
  useMmap: true,
  nGpuLayers: 0
};

const llama = new LLM(LLamaCpp);


llama.load(config).then(handler);

async function handler() {
  const data = await loader.load();
  const textSplitter = new RecursiveCharacterTextSplitter();
  const embeddings = new LLamaEmbeddings({ maxConcurrency: 1 }, llama);
  const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);

Question: {question}
Helpful Answer:`;

  const chain = new RetrievalQAChain({
    combineDocumentsChain: loadQAStuffChain(model, {
      prompt: new PromptTemplate({
        inputVariables: ['context', 'question'],
        template
      })
    }),
    retriever: vectorStore.asRetriever(),
    returnSourceDocuments: true,
    inputKey: 'question'
  });

  console.log('Write a question:');

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
  });

  rl.on('line', async (query) => {
    const response = await chain.call({ query });

    console.log(response);
  });
}
