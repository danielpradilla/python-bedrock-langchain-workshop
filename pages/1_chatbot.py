import sys
import os
import streamlit as st

import chatbot_lib as glib
import common_functions as cf

DEFAULT_MODEL_ID = 'anthropic.claude-v2'


st.set_page_config(page_title="Chatbot") #HTML title
st.title("Chatbot") #page title

if 'memory' in st.session_state:
    del st.session_state['memory']
if 'chat_history' in st.session_state:
    del st.session_state['chat_history']
    
st.write(f"This is a basic chatbot app using the {DEFAULT_MODEL_ID} model")

bedrock_client, bedrock_runtime_client = cf.get_clients()

if 'chat_history' not in st.session_state: #see if the chat history hasn't been created yet
    st.session_state.chat_history = [] #initialize the chat history

#Re-render the chat history (Streamlit re-runs this script, so need this to preserve previous chat messages)
for message in st.session_state.chat_history: #loop through the chat history
    with st.chat_message(message["role"]): #renders a chat line for the given role, containing everything in the with block
        st.markdown(message["text"]) #display the chat content


input_text = st.chat_input("Chat with your bot here") #display a chat input box


if input_text: #run the code in this if block after the user submits a chat message
    
    with st.chat_message("user"): #display a user chat message
        st.markdown(input_text) #renders the user's latest message
    
    st.session_state.chat_history.append({"role":"user", "text":input_text}) #append the user's latest message to the chat history
    
    chat_response = glib.get_chat_response_basic(model_id=DEFAULT_MODEL_ID, input_text=input_text) #call the model through the supporting library
    
    with st.chat_message("assistant"): #display a bot chat message
        st.markdown(chat_response) #display bot's latest response
    
    st.session_state.chat_history.append({"role":"assistant", "text":chat_response}) #append the bot's latest message to the chat history

