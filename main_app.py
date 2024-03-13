import streamlit as st
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    layout="wide"
)

st.write("# 👋 🛏️🪨 🦜🔗")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Hi
    This is an intro to leveraging LangChain with foundational models provided by AWS Bedrock, enabling you to explore the cutting-edge capabilities of language models without suffering (much)
  
    👈 Select a demo from the sidebar
    
    ### Docs
    - [LangChain](https://python.langchain.com/docs/get_started/introduction)
    - [Building with Amazon Bedrock and LangChain](https://catalog.workshops.aws/building-with-amazon-bedrock/en-US) (heavily "inspired" by this)
    - [Bedrock Playground](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/chat-playground)
    - [streamlit.io](https://streamlit.io)
"""
)


    
