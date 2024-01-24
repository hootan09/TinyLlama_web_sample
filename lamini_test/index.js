    class MyClassificationPipeline {
        static task = 'text2text-generation';
        static model = 'Xenova/LaMini-T5-61M';
        static instance = null;

        static async getInstance(progress_callback = null) {
            if (this.instance === null) {
                // Dynamically import the Transformers.js library
                let { pipeline, env } = await import('@xenova/transformers');

                // NOTE: Uncomment this to change the cache directory
                env.cacheDir = './lamini_model';

                this.instance = pipeline(this.task, this.model, { progress_callback });
            }

            return this.instance;
        }
    }

    (async()=>{
        // Comment out this line if you don't want to start loading the model as soon as the server starts.
        // If commented out, the model will be loaded when the first request is received (i.e,. lazily).
        await MyClassificationPipeline.getInstance(progress_callback = async(x)=> {
            console.clear();
            console.log(`Loading Model Status ===> ${x?.status}`);
        });

        const model = await MyClassificationPipeline.getInstance();

        let input_prompt = 'Once upon a time, a man who lived in'
        // Generate a response
        const result = await model(input_prompt,max_length=512, do_sample=true,{ 
            // callback_function: async(x) => {
            //     // let chunked = model.tokenizer.decode(x[0].output_token_ids, { skip_special_tokens: true })
            //     // console.log(x);
            // }
        });
        console.log(result[0]['generated_text']);
})()

