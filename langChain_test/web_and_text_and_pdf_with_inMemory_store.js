const { removeWordsFromStartAndAfter } = require('../utils/utils');
(async()=> {
    ////web loader
    let { CheerioWebBaseLoader } = await import("langchain/document_loaders/web/cheerio");
    const loader = new CheerioWebBaseLoader(
    "https://cv.nikitv.ir"
    );
    const data = await loader.load();

    // console.log(data);

    ////or text loader
    // let { TextLoader } = await  import("langchain/document_loaders/fs/text");
    // const loader = new TextLoader("./example.txt");
    // const data = await loader.load();


    ////or pdf
    // const parse = require("pdf-parse");
    // let { PDFLoader } =  await import("langchain/document_loaders/fs/pdf");
    // const loader = new PDFLoader("./data/MachineLearning-Lecture01.pdf");
    // const data = await loader.load();

    let { RecursiveCharacterTextSplitter } = await  import("langchain/text_splitter");
    const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 0,
    });

    const splitDocs = await textSplitter.splitDocuments(data);

    // console.log(splitDocs);

    let { MemoryVectorStore } = await  import("langchain/vectorstores/memory");
    let { env, pipeline } = await import('@xenova/transformers');
    let { HuggingFaceTransformersEmbeddings } = await import("@langchain/community/embeddings/hf_transformers");
    
    env.cacheDir = './all-MiniLM';
    const embeddings = new HuggingFaceTransformersEmbeddings({
        modelName: "Xenova/all-MiniLM-L6-v2",
    });

    const vectorStore = await MemoryVectorStore.fromDocuments(
        splitDocs,
        embeddings
    );

    let question = "what skills mohammad nikravesh have?";
    const relevantDocs = await vectorStore.similaritySearch(question, 3);
    // console.log(relevantDocs.length);
    // console.log(relevantDocs[0]);
    let context = relevantDocs.map((item) => item?.pageContent).join('\n');
    
    // console.log(context);

    const template = `You are a friendly assistant.Use the following data to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    keep the answer as concise as possible.
    data is:${context}`;
  //  console.log(template);



  class MyClassificationPipeline {
    static task = 'text-generation';
    static model = 'Xenova/TinyLlama-1.1B-Chat-v1.0';
    static instance = null;

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            this.instance = pipeline(this.task, this.model, { progress_callback });
        }

        return this.instance;
    }
}

  // Comment out this line if you don't want to start loading the model as soon as the server starts.
  // If commented out, the model will be loaded when the first request is received (i.e,. lazily).
  MyClassificationPipeline.getInstance(progress_callback = async(x)=> {
      console.clear();
      console.log(`Loading Model Status ===> ${x?.status}`);
  });

  const model = await MyClassificationPipeline.getInstance();

  // Define the list of messages
  const messages = [
    { "role": "system", "content": context },
    { "role": "user", "content": question },
]

// Construct the prompt
const prompt = model.tokenizer.apply_chat_template(messages, {
    tokenize: false, add_generation_prompt: true,
});

// Generate a response
const result = await model(prompt, {
    max_new_tokens: 256,
    temperature: 0.7,
    do_sample: true,
    top_k: 50,
    callback_function: async(x) => {
        let chunked = model.tokenizer.decode(x[0].output_token_ids, { skip_special_tokens: true })
        console.log(removeWordsFromStartAndAfter(chunked, '<|assistant|>\n'));
    }
});

      
})();