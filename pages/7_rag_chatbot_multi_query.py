import sys
import os
import streamlit as st

from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

import common_functions as cf
import rag_chatbot_lib as glib


st.set_page_config(page_title="RAG Chatbot with Multiple Queries") #HTML title
st.title("RAG Chatbot with Multiple Queries") #page title

#st.write("Source files:" + glib.DOCUMENTS_PATH)

bedrock_client, bedrock_runtime_client = cf.get_clients()

models = cf.get_models(bedrock_client)
default_model_key = next((key for key, value in models.items() if value == glib.DEFAULT_MODEL_ID), None)
default_model_index = list(models.keys()).index(default_model_key) if default_model_key else 0
if models:
    # Convert model names and IDs into a sorted list (optional but can improve usability)
    sorted_model_names = sorted(models.keys())

select_model = st.selectbox(
        "Select model",
        list(models.keys()),
        key="model",
        index=default_model_index
)
selected_model_id = models[select_model]


documents_paths = glib.DOCUMENTS_PATHS_LIST
default_documents_path_key = next((key for key, value in documents_paths.items() if value == glib.DEFAULT_DOCUMENTS_PATH), None)
default_documents_index = list(documents_paths.keys()).index(default_documents_path_key) if default_documents_path_key else 0

select_index = st.selectbox(
        "Select index",
        list(documents_paths.keys()),
        key="source",
        index=default_documents_index
)

if select_index:
    #Add the vector index to the session cache.
    #This allows us to maintain an in-memory vector database per user session.
    selected_documents_path = documents_paths[select_index]
    with st.spinner("Indexing documents..."): #show a spinner while the code in this with block runs
        st.session_state.vector_index = glib.get_index(selected_documents_path) #retrieve the index through the supporting library and store in the app's session cache


#Add the LangChain memory to the session cache.

#This allows us to maintain a unique chat memory per user session. Otherwise, the chatbot won't be able to remember past messages with the user.
#In Streamlit, session state is tracked server-side. If the browser tab is closed, or the application is stopped, the session and its chat history will be lost. In a real-world application, you would want to track the chat history in a database like Amazon DynamoDB 

if 'memory' not in st.session_state: #see if the memory hasn't been created yet
    st.session_state.memory = glib.get_memory() #initialize the memory

#Add the UI chat history to the session cache.

#This allows us to re-render the chat history to the UI as the Streamlit app is re-run with each user interaction. Otherwise, the old messages will disappear from the user interface with each new chat message.

if 'chat_history' not in st.session_state: #see if the chat history hasn't been created yet
    st.session_state.chat_history = [] #initialize the chat history



#Add the for loop to render previous chat messages.
#Re-render the chat history (Streamlit re-runs this script, so need this to preserve previous chat messages)
for message in st.session_state.chat_history: #loop through the chat history
    with st.chat_message(message["role"]): #renders a chat line for the given role, containing everything in the with block
        st.markdown(message["text"]) #display the chat content


input_text = st.chat_input("Chat with your bot here") #display a chat input box

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

if input_text: #run the code in this if block after the user submits a chat message
    with st.chat_message("user"): #display a user chat message
        st.markdown(input_text) #renders the user's latest message
    st.session_state.chat_history.append({"role":"user", "text":input_text}) #append the user's latest message to the chat history

    with st.chat_message("assistant"): #display a bot chat message
        st_callback = StreamHandler(st.empty())
        chat_response = glib.get_multiquery_rag_chat_streaming_response(model_id=selected_model_id, input_text=input_text, memory=st.session_state.memory, index=st.session_state.vector_index,streaming_callback=st_callback) #call the model through the supporting library
        #st.markdown(chat_response) #display bot's latest response  
        if chat_response["source_documents"][0]:
            st.write(chat_response["source_documents"][0])
#            for document in chat_response["source_documents"]:
#                st.write(document.metadata['source'])
        st.session_state.chat_history.append({"role":"assistant", "text":chat_response['answer']}) #append the bot's latest message to the chat history
