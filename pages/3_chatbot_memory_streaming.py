import sys
import os
import streamlit as st

from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

import chatbot_lib as glib
import common_functions as cf

DEFAULT_MODEL_ID = 'anthropic.claude-v2'

st.set_page_config(page_title="Chatbot") #HTML title
st.title("Chatbot") #page title

st.write(f"This is a basic chatbot app using the {DEFAULT_MODEL_ID} model, with memory and streaming")

bedrock_client, bedrock_runtime_client = cf.get_clients()

if 'memory' not in st.session_state: #see if the memory hasn't been created yet
    st.session_state.memory = glib.get_memory(model_id=DEFAULT_MODEL_ID) #initialize the memory

if 'chat_history' not in st.session_state: #see if the chat history hasn't been created yet
    st.session_state.chat_history = [] #initialize the chat history

#Re-render the chat history (Streamlit re-runs this script, so need this to preserve previous chat messages)
for message in st.session_state.chat_history: #loop through the chat history
    with st.chat_message(message["role"]): #renders a chat line for the given role, containing everything in the with block
        st.markdown(message["text"]) #display the chat content


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
        


input_text = st.chat_input("Chat with your bot here") #display a chat input box

if input_text: #run the code in this if block after the user submits a chat message
    
    with st.chat_message("user"): #display a user chat message
        st.markdown(input_text) #renders the user's latest message
    
    st.session_state.chat_history.append({"role":"user", "text":input_text}) #append the user's latest message to the chat history

    with st.chat_message("assistant"): #display a bot chat message
        st_callback = StreamHandler(st.empty())
        chat_response = glib.get_chat_response(model_id=DEFAULT_MODEL_ID, input_text=input_text, memory=st.session_state.memory, streaming_callback=st_callback) #call the model through the supporting library
                
        st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) #append the bot's latest message to the chat history

