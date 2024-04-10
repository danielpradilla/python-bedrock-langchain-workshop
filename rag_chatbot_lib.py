import os
import glob
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import Bedrock
from langchain.chains import ConversationalRetrievalChain

from langchain_community.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.document_loaders import TextLoader

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


import common_functions as cf

DEFAULT_DOCUMENTS_PATH = '../datasets/pdf/digital_corpora/3003'
DEFAULT_MODEL_ID = 'anthropic.claude-v2'
DOCUMENTS_PATHS_LIST = {"Digital Corpora PDFs Folder 3003":"../datasets/pdf/digital_corpora/3003", 
                        "United Nations Parallel Corpus 1.0 - 2010 HRC": "../datasets/UNv1.0-TEI/en/2010/a/hrc/14"}

def generate_index_path(documents_path):
    # Remove initial '../' or './' if present
    clean_path = documents_path.lstrip('./').lstrip('../')
    # Replace '/' with '_'
    index_path_suffix = clean_path.replace('/', '_')
    return f'faiss_index_{index_path_suffix}'

def get_llm(model_id, streaming_callback):
    model_kwargs = cf.get_inference_parameters(model_id)

    llm = Bedrock(
        region_name=cf.get_region(),
        model_id=model_id,
        streaming=True,
        model_kwargs=model_kwargs,
        callbacks=[streaming_callback]
    )
    
    return llm


#function to create an in-memory vector store
def get_index(documents_path):
    embeddings_client = BedrockEmbeddings(
        region_name=cf.get_region()
    )

    index_path = generate_index_path(documents_path)
    if not os.path.exists(index_path):
        documents = []
        for file_path in glob.glob(os.path.join(documents_path, '**'), recursive=True):
            print(file_path)
            try:
                if file_path.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.endswith('.docx') or file_path.endswith('.doc'):
                    loader = Docx2txtLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                elif file_path.endswith('.xml'):
                    #loader = UnstructuredXMLLoader(file_path, mode="elements", strategy="fast")
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        text_splitter = RecursiveCharacterTextSplitter( #create a text splitter
            separators=["\n\n", "\n", ".", " "], #split chunks at (1) paragraph, (2) line, (3) sentence, or (4) word, in that order
            chunk_size=1000, #divide into 1000-character chunks using the separators above
            chunk_overlap=100 #number of characters that can overlap with previous chunk
        )
        split_documents = text_splitter.split_documents(documents)
        index = FAISS.from_documents(split_documents, embeddings_client)
        index.save_local(index_path)
    else:
        index = FAISS.load_local(index_path, embeddings_client, allow_dangerous_deserialization=True)

    return index #return the index to be cached by the client app


def get_memory(): #create memory for this chat session
    
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, output_key='answer') #Maintains a history of previous messages
    
    return memory

def get_rag_chat_streaming_response(model_id, input_text, memory, index, streaming_callback):

    llm = get_llm(model_id, streaming_callback)
    
    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm, index.as_retriever(), memory=memory, return_source_documents=True)
    
    chat_response = conversation_with_retrieval({"question": input_text}) #pass the user message, history, and knowledge to the model

    return chat_response


def get_multiquery_rag_chat_streaming_response(model_id, input_text, memory, index, streaming_callback):

    llm = get_llm(model_id, streaming_callback)
    
    prompt_template = """You are an AI language model assistant.
        Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database.
        By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations  of distance-based similarity search.
        Provide these alternative questions separated by newlines.
        Original question: {question}
        """
    
    PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["question"]
        )

    multiquery_retriever = MultiQueryRetriever.from_llm(llm=llm, retriever=index.as_retriever(), prompt=PROMPT, parser_key="lines") #custom prompt parsing questions in lines

    conversation_with_retrieval = ConversationalRetrievalChain.from_llm(llm=llm, retriever=multiquery_retriever, memory=memory, return_source_documents=True)
    
    chat_response = conversation_with_retrieval({"question": input_text}) #pass the user message, history, and knowledge to the model

    return chat_response


def get_mmr_rag_chat_streaming_response(model_id, input_text, memory, index, streaming_callback):

    llm = get_llm(model_id, streaming_callback)

    similarity_retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": 8})
    similarity_docs = similarity_retriever.get_relevant_documents(input_text)
    mmr_retriever = index.as_retriever(search_type="mmr", search_kwargs={"k": 8})
    mmr_docs = mmr_retriever.get_relevant_documents(input_text)

    similarity_docs_contents = {getattr(doc, 'page_content') for doc in similarity_docs}
    mmr_docs_contents = {getattr(doc, 'page_content') for doc in mmr_docs}

    unique_to_mmr = [doc for doc in mmr_docs if getattr(doc, 'page_content') in mmr_docs_contents - similarity_docs_contents]
    print("unique in mmr:")
    for doc in unique_to_mmr:
        print(doc)

    conversation_with_retrieval_mmr = ConversationalRetrievalChain.from_llm(llm, mmr_retriever, memory=memory, return_source_documents=True)
    
    chat_response = conversation_with_retrieval_mmr({"question": input_text}) #pass the user message, history, and knowledge to the model

    return chat_response


def get_rag_chat_with_contextual_compression_streaming_response(model_id, input_text, memory, index, streaming_callback):

    llm = get_llm(model_id, streaming_callback)

    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=index.as_retriever(search_kwargs={"k": 2}))

    conversation_with_compressor_retrieval = ConversationalRetrievalChain.from_llm(llm, compression_retriever, memory=memory, return_source_documents=True)
    
    chat_response = conversation_with_compressor_retrieval({"question": input_text}) #pass the user message, history, and knowledge to the model

    return chat_response