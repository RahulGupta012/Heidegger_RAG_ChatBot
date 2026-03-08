This is a RAG-based architecture including data loading and scraping, text preprocessing, tokenization, embedding generation, vector-semantic search, 
and storage. It has a knowledge base of a philosopher who hates technology and has harsh views on it.

Good prompt engineering has been performed here to make the responses very specific and accurate. 
You will see that there are not even minor hallucinations.

I used the Cohere LLM as the thinking model. I also provided an interactive chatbot-based graphical interface for users. 

To run the interface, execute the frontend.py file.

You required your own LLM api key for runnig the files. You can just grab it from Cohere Website free of cost.
when you get it just put it in the .env file at the "API_KEY" varibale.

and you are all set to use this.
