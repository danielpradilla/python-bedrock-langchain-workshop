import os
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate

import common_functions as cf

def get_llm(model_id, temperature):
    
    model_kwargs = cf.get_inference_parameters(model_id)
    model_kwargs['temperature'] = temperature

    llm = Bedrock(
        region_name=cf.get_region(),
        model_id=model_id,
        model_kwargs=model_kwargs,
    )    
    return llm


def read_file(file_name):
    with open(file_name, "r") as f:
        text = f.read()
     
    return text


def get_context_list():
    return ["Prompt engineering basics", "Content creation", "Summarization", "Question and answer", "Translation", "Analysis: Positive email", "Analysis: Negative email", "Code", "Advanced techniques: Claude"]


def get_context(lab):
    if lab == "Prompt engineering basics":
        return read_file("basics.txt")
    if lab == "Summarization":
        return read_file("summarization_content.txt")
    elif lab == "Question and answer":
        return read_file("qa.txt")
    elif lab == "Analysis: Positive email":
        return read_file("analysis_positive.txt")
    elif lab == "Analysis: Negative email":
        return read_file("analysis_negative.txt")
    elif lab == "Content creation":
        return read_file("qa.txt")
    elif lab == "Translation":
        return read_file("qa.txt")
    elif lab == "Code":
        return ""
    elif lab == "Advanced techniques: Claude":
        return read_file("summarization_content.txt")


def get_prompt(template, context=None, user_input=None):
    
    prompt_template = PromptTemplate.from_template(template) #this will automatically identify the input variables for the template
    
    if "{context}" not in template:
        prompt = prompt_template.format()
    else:
        prompt = prompt_template.format(context=context) #, user_input=user_input)
    
    return prompt



def get_text_response(model_id, temperature, template, context=None, user_input=None): #text-to-text client function
    llm = get_llm(model_id, temperature)
    
    prompt = get_prompt(template, context, user_input)
    
    response = llm.predict(prompt) #return a response to the prompt

    print(response)
    
    return response
