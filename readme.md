# Bedrock + Langchain workshop playground

A streamlit app with a few examples on how to use AWS Bedrock and LangChain to develop quick PoCs

## Installing dependencies
    pip install -r requirements.txt  

## How to run
    streamlit run main_app.py --server.port 8080 --server.headless true   

## "Organization" of code
All back end is in _lib.py files in the root. Front end stuff is under "pages".

The rag_chatbot_lib has a variable called DEFAULT_DOCUMENTS_PATH that should point to were the raw datasets are stored. Other globals in the same file point to the locaton of the FAISS indexes to be used.

The notebook folder contains a few notebooks to test or generate data. It's better if you generate your indexes beforehand, otherwise you'll die waiting.

## A few prompts to test with the Agent demo
- What was the latest result of Real Madrid?
- What is the current time and temperature in <Your_City>?
- What does wikipedia say about the recent events in <Your_Country>?
- Find me some recent youtube videos about <Topic>. Make the links clickable.



## Sourcing the RAG data
 - [Digital Corpora PDFs](https://corp.digitalcorpora.org/corpora/files/CC-MAIN-2021-31-PDF-UNTRUNCATED/)
 - [United Nations Parallel Corpus](https://conferences.unite.un.org/UNCorpus/)
 - [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)

### Docs
- [LangChain](https://python.langchain.com/docs/get_started/introduction)
- [Building with Amazon Bedrock and LangChain](https://catalog.workshops.aws/building-with-amazon-bedrock/en-US) (heavily "inspired" by this)
- [Bedrock Playground](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/chat-playground)
- [streamlit.io](https://streamlit.io)
    