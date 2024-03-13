import streamlit as st
import prompt_playground_lib as glib
st.set_page_config(layout="wide")

import common_functions as cf

DEFAULT_MODEL_ID = 'anthropic.claude-v2'
INITIAL_TEMPERATURE = 0.0

bedrock_client, bedrock_runtime_client = cf.get_clients()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Context")
    context_text = st.text_area("Context text:", value="", height=350)

with col2:
    
    st.subheader("Prompt & model")
    
    prompt_text = st.text_area("Prompt template text:", height=350)

    models = cf.get_models(bedrock_client)
    default_model_key = next((key for key, value in models.items() if value == DEFAULT_MODEL_ID), None)
    default_model_index = list(models.keys()).index(default_model_key) if default_model_key else 0
    if models:
        # Convert model names and IDs into a sorted list (optional but can improve usability)
        sorted_model_names = sorted(models.keys())

    select_model = st.selectbox(
            "Select model",
            list(models.keys()),
            key="visibility",
            index=default_model_index
    )
    selected_model_id = models[select_model]
    
    #selected_temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    
    # temperature = st.slider("Temperature", min_value=0.0,
    #                     max_value=1.0, value=INITIAL_TEMPERATURE, step=0.1)

    process_button = st.button("Run", type="primary")


with col3:
    st.subheader("Result")
    
    if process_button:
        with st.spinner("Running..."):
            response_content = glib.get_text_response(model_id=selected_model_id, temperature=INITIAL_TEMPERATURE, template=prompt_text, context=context_text)
            st.write(response_content)


