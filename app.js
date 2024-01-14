const { removeWordsFromStartAndAfter, embedQuery } = require('./utils/utils');
const express = require('express');
const app = express();
const expressWs = require('express-ws')(app);
const port = process.env.Port || 3000;

app.use(express.static('public'))
// set the view engine to ejs
app.set('view engine', 'ejs');


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
MyClassificationPipeline.getInstance(progress_callback = async(x)=> {
    console.clear();
    console.log(`Server is listening on port ${port}\n`);
    console.log(`Loading Model Status ===> ${x?.status}`);
});

/*
 Single sapi call request sample
*/
app.get('/askchat', async (req, res) => {
    const {text} = req.query;
    const model = await MyClassificationPipeline.getInstance();

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
        // callback_function: async(x) => {
        //     let chunked = model.tokenizer.decode(x[0].output_token_ids, { skip_special_tokens: true })
        //     console.log(removeWordsFromStartAndAfter(chunked, '<|assistant|>\n'));
        // }
    });

    // console.log(result);

    res.statusCode = 200;
    // Send the JSON response
    let generatedTextFromModel = removeWordsFromStartAndAfter(result?.[0]['generated_text'], '<|assistant|>\n')
    res.send(generatedTextFromModel);
});

app.ws('/streamChat', async(ws, req) => {
    ws.on('message', async(text) => {
        const model = await MyClassificationPipeline.getInstance();
    
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
            callback_function: async(x) => {
                let chunked = model.tokenizer.decode(x[0].output_token_ids, { skip_special_tokens: true })
                // console.log(removeWordsFromStartAndAfter(chunked, '<|assistant|>\n'));
                ws.send(removeWordsFromStartAndAfter(chunked, '<|assistant|>\n'));
            }
        });
    });
});

app.get('/embeding/:text', async(req,res)=> {
    let {text} = req.params;
    // console.log(text);
    let result = await embedQuery(text);
    res.send(JSON.stringify({result: result}));
})



// use res.render to load up an ejs view file

// index page
app.get('/', (req, res) => {
    var mascots = [
        { name: 'Sammy', organization: "DigitalOcean", birth_year: 2012 },
        { name: 'Tux', organization: "Linux", birth_year: 1996 },
        { name: 'Moby Dock', organization: "Docker", birth_year: 2013 }
    ];
    var tagline = "No programming concept is complete without a cute animal mascot.";

    res.render('pages/index', {
        mascots: mascots,
        tagline: tagline
    });
});

// about page
app.get('/about', (req, res) => {
    res.render('pages/about');
});

// chat page
app.get('/chat', (req, res) => {
    res.render('pages/chat');
});

app.listen(port);
console.log(`Server is listening on port ${port}`);