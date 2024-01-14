

(async()=> {
    let { env } = await import('@xenova/transformers');
    env.cacheDir = './all-MiniLM';
    let { HuggingFaceTransformersEmbeddings } = await import("@langchain/community/embeddings/hf_transformers");
    
    const model = new HuggingFaceTransformersEmbeddings({
      modelName: "Xenova/all-MiniLM-L6-v2",
    });

    const res = await model.embedQuery(
      "What would be a good company name for a company that makes colorful socks?"
    );
    console.log({ res });
    /* Embed documents */
    const documentRes = await model.embedDocuments(["Hello world", "Bye bye"]);
    console.log({ documentRes });
})();
