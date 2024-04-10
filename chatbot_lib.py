import os
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.chains import ConversationChain

import common_functions as cf


def get_llm_basic(model_id):
    model_kwargs = cf.get_inference_parameters(model_id)

    llm = Bedrock(
        region_name=cf.get_region(),
        model_id=model_id,
        model_kwargs=model_kwargs
    )
    
    return llm

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

def get_chat_llm(model_id, streaming_callback):
    model_kwargs = cf.get_inference_parameters(model_id)

    llm = BedrockChat(
        region_name=cf.get_region(),
        model_id=model_id,
        streaming=True,
        model_kwargs=model_kwargs,
        callbacks=[streaming_callback]
    )
    
    return llm

"""
Add a function to initialize a LangChain memory object.
In this case, we are using the ConversationSummaryBufferMemory class. 
This allows us to track the most recent messages and summarize older messages 
so that the chat context is maintained over a long conversation.
"""

def get_memory(model_id): #create memory for this chat session
    
    #ConversationSummaryBufferMemory requires an LLM for summarizing older messages
    #this allows us to maintain the "big picture" of a long-running conversation
    llm = get_llm_basic(model_id)
    
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1024) #Maintains a summary of previous messages
    
    return memory

def get_chat_response_basic(model_id, input_text): #chat client function
    
    llm = get_llm_basic(model_id)
    
    conversation_with_summary = ConversationChain( #create a chat client
        llm = llm, #using the Bedrock LLM
        verbose = True #print out some of the internal states of the chain while running
    )
    
    chat_response = conversation_with_summary.predict(input=input_text) #pass the user message and summary to the model
    
    return chat_response


def get_chat_response(model_id, input_text, memory, streaming_callback): #chat client function
    
    llm = get_llm(model_id, streaming_callback)
    
    conversation_with_summary = ConversationChain( #create a chat client
        llm = llm, #using the Bedrock LLM
        memory = memory, #with the summarization memory
        verbose = True #print out some of the internal states of the chain while running
    )
    
    chat_response = conversation_with_summary.predict(input=input_text) #pass the user message and summary to the model
    
    return chat_response

