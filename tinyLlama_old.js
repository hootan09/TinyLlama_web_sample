
const http = require('http');
const querystring = require('querystring');
const url = require('url');


class MyClassificationPipeline {
  static task = 'text-generation';
  static model = 'Xenova/TinyLlama-1.1B-Chat-v1.0';
  static instance = null;

  static async getInstance(progress_callback = null) {
    if (this.instance === null) {
      // Dynamically import the Transformers.js library
      let { pipeline, env } = await import('@xenova/transformers');

      // NOTE: Uncomment this to change the cache directory
      env.cacheDir = './TinyLlama';

      this.instance = pipeline(this.task, this.model, { progress_callback });
    }

    return this.instance;
  }
}

// Comment out this line if you don't want to start loading the model as soon as the server starts.
// If commented out, the model will be loaded when the first request is received (i.e,. lazily).
MyClassificationPipeline.getInstance();

// Define the HTTP server
const server = http.createServer();
const hostname = '127.0.0.1';
const port = 3000;

// Listen for requests made to the server
server.on('request', async (req, res) => {
  const model = await MyClassificationPipeline.getInstance(); 
  // Parse the request URL
  const parsedUrl = url.parse(req.url);

  // Extract the query parameters
  const { text } = querystring.parse(parsedUrl.query);

  // Set the response headers
  res.setHeader('Content-Type', 'application/json');

  let response;
  if (parsedUrl.pathname === '/tinyllama' && text) {
    
    
    // Define the list of messages
    const messages = [
      { "role": "system", "content": "You are a friendly assistant." },
      { "role": "user", "content": text },
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
    });

    console.log(result);

    res.statusCode = 200;
    // Send the JSON response
    let generatedTextFromModel = removeWordsFromStartAndAfter(result?.[0]['generated_text'], '<|assistant|>\n')
    res.end(JSON.stringify(generatedTextFromModel));
} else {
    response = { 'error': 'Bad request' }
    res.statusCode = 400;
    res.end(JSON.stringify(response))
}

});

server.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
});

//http://localhost:3000/gpt2lamini?text=hello

function removeWordsFromStartAndAfter(inputString, targetWord) {
    const targetIndex = inputString.indexOf(targetWord);
        if (targetIndex !== -1) {
            const wordsToKeep = inputString.substring(targetIndex+ 13).trim();
            return wordsToKeep;
        }
        return inputString;
}