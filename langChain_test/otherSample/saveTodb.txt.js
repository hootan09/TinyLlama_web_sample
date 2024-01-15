const { OpenAIEmbeddings } = require("langchain/embeddings/openai");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { HNSWLib } = require("langchain/vectorstores/hnswlib");
const fs = require("fs");

const OPENAI_API_KEY = "YOUR_API_KEY_GOES_HERE";

async function generateAndStoreEmbeddings() {

  // STEP 1: Load the data
  const trainingText = fs.readFileSync("training-data.txt", "utf8");

  // STEP 2: Split the data into chunks
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
  });

  // STEP 3: Create documents
  const docs = await textSplitter.createDocuments([trainingText]);

  // STEP 4: Generate embeddings from documents
  const vectorStore = await HNSWLib.fromDocuments(
    docs,
    new OpenAIEmbeddings({ openAIApiKey: OPENAI_API_KEY }),
  );

  // STEP 5: Save the vector store
  vectorStore.save("hnswlib");
}