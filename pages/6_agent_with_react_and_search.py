import sys
import os
import streamlit as st

from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

import agent_lib as glib
import common_functions as cf

DEFAULT_MODEL_ID = 'anthropic.claude-v2'
DEFAULT_PROMPT_URL = 'hwchase17/react'

st.set_page_config(page_title="Agent ReAct and search") #HTML title
st.title("Agent ReAct and Search") #page title

st.markdown(f"This is an agent app using multiple tools")

bedrock_client, bedrock_runtime_client = cf.get_clients()

models = cf.get_models(bedrock_client)
default_model_key = next((key for key, value in models.items() if value == DEFAULT_MODEL_ID), None)
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



if 'memory' not in st.session_state: #see if the memory hasn't been created yet
    st.session_state.memory = glib.get_memory() #initialize the memory

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
        chat_response = glib.get_agent_streaming_response(agent_type='react', model_id=DEFAULT_MODEL_ID, input_text=input_text, prompt_url=DEFAULT_PROMPT_URL, streaming_callback=st_callback) #call the model through the supporting library
        st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) #append the bot's latest message to the chat history

