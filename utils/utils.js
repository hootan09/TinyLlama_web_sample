const removeWordsFromStartAndAfter = (inputString, targetWord) => {
    const targetIndex = inputString.indexOf(targetWord);
        if (targetIndex !== -1) {
            const wordsToKeep = inputString.substring(targetIndex+ 13).trim();
            return wordsToKeep;
        }
        return inputString;
}

/**
 * 
 * @param {string} Docs 
 * @returns [number]
 */
const embedQuery = async(text='')=> {
    let { env } = await import('@xenova/transformers');
    env.cacheDir = './tinyLlama';
    let { HuggingFaceTransformersEmbeddings } = await import("@langchain/community/embeddings/hf_transformers");
    
    const model = new HuggingFaceTransformersEmbeddings({
      modelName: "Xenova/all-MiniLM-L6-v2",
    });

    const res = await model.embedQuery(text);
    return res;
}

/**
 * 
 * @param {Array.<string>} Docs 
 * @returns [ [number],.. ]
 */
const embedDocuments = async(Docs = [])=> {
    let { env } = await import('@xenova/transformers');
    env.cacheDir = './tinyLlama';
    let { HuggingFaceTransformersEmbeddings } = await import("@langchain/community/embeddings/hf_transformers");
    
    const model = new HuggingFaceTransformersEmbeddings({
      modelName: "Xenova/all-MiniLM-L6-v2",
    });
    const documentRes = await model.embedDocuments(Docs);
    return documentRes;
}

module.exports = {
    removeWordsFromStartAndAfter,
    embedQuery,
    embedDocuments
}