import os
import boto3
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_REGION_NAME = os.getenv("AWS_REGION")

# Decorator to cache the resource initialization
@st.cache_resource
def get_boto3_session():
    # Check if session is already set in `st.session_state`
    if "aws_boto3_session" not in st.session_state:
        # Retrieve AWS credentials from environment variables
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.getenv("AWS_SESSION_TOKEN")  # This might be optional in some cases
        region_name = DEFAULT_REGION_NAME

        # Initialize a new session with the retrieved credentials
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,  # This can be omitted if not using temporary credentials
            region_name=region_name
        )
        
        # Store the session in `st.session_state` for future use
        st.session_state["aws_boto3_session"] = session
    else:
        # Retrieve the session from `st.session_state`
        session = st.session_state["aws_boto3_session"]

    return session

@st.cache_resource
def get_clients():
    session = get_boto3_session()
    # Create clients for the services you want to use
    bedrock_client = session.client(service_name="bedrock", region_name=DEFAULT_REGION_NAME)
    bedrock_runtime_client = session.client(service_name="bedrock-runtime", region_name=DEFAULT_REGION_NAME)
    
    return bedrock_client, bedrock_runtime_client

def get_region():
    session = get_boto3_session()
    return session.region_name

@st.cache_resource
def get_models(_client):
    rtn ={}
    response = _client.list_foundation_models(
        byOutputModality='TEXT',
        byInferenceType='ON_DEMAND'
    )
    for model in response['modelSummaries']:
        rtn[f"{model['modelName']} ({model['modelId']})"]=model['modelId']
    return rtn




def get_inference_parameters(model_id): #return a default set of parameters based on the model's provider
    bedrock_model_provider = model_id.split('.')[0] #grab the model provider from the first part of the model id
    
    if (bedrock_model_provider == 'anthropic'): #Anthropic model
        if "anthropic.claude-3-" in model_id:
            return { 
                "max_tokens": 2048,
                "temperature": 0.0,
                "top_k": 250,
                "top_p": 1,
                "stop_sequences": ["\n\nHuman"],
            }
        else:
            return { #anthropic
                "max_tokens_to_sample": 512,
                "temperature": 0, 
                "top_k": 250, 
                "top_p": 1, 
                "stop_sequences": ["\n\nHuman:"]
            }
    
    
    elif (bedrock_model_provider == 'ai21'): #AI21
        return { #AI21
            "maxTokens": 512, 
            "temperature": 0, 
            "topP": 0.5, 
            "stopSequences": [], 
            "countPenalty": {"scale": 0 }, 
            "presencePenalty": {"scale": 0 }, 
            "frequencyPenalty": {"scale": 0 } 
           }
    
    elif (bedrock_model_provider == 'cohere'): #COHERE
        return {
            "max_tokens": 512,
            "temperature": 0,
            "p": 0.01,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE"
        }
    
    elif (bedrock_model_provider == 'meta'): #META
        return {
            "temperature": 0,
            "top_p": 0.9,
            "max_gen_len": 512
        }

    elif (bedrock_model_provider == 'mistral'): #Mistral
        return {
            "max_tokens": 512,
            "temperature": 0,
            "top_p": 0.7,
            "top_k": 50
           }
    
    else: #Amazon
        #For the LangChain Bedrock implementation, these parameters will be added to the 
        #textGenerationConfig item that LangChain creates for us
        return { 
            "maxTokenCount": 512, 
            "stopSequences": [], 
            "temperature": 0, 
            "topP": 0.9 
        }
   