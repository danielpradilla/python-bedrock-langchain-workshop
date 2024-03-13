from langchain import hub
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent
from langchain.agents import AgentType, initialize_agent, load_tools

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.tools import YouTubeSearchTool

from langchain_community.llms import Bedrock

import common_functions as cf


def get_tools():
    tools = [TavilySearchResults(max_results=1, api_wrapper=TavilySearchAPIWrapper())]
    return tools

def get_tools2(llm):
    tools = load_tools(["google-serper","wikipedia"], llm=llm)
    tools.append(YouTubeSearchTool())
    return tools

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

def get_prompt(prompt_url):
    prompt = hub.pull(prompt_url)
    return prompt

def get_memory(): #create memory for this chat session
    
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, output_key='output') #Maintains a history of previous messages
    
    return memory


def get_agent_streaming_response(agent_type, model_id, input_text, prompt_url, streaming_callback):

    llm = get_llm(model_id, streaming_callback)
    tools = get_tools2(llm)
    prompt = get_prompt(prompt_url)

    if agent_type == 'react':
        #get ReAct agent
        agent = create_react_agent(llm, tools, prompt)
    elif agent_type == 'structured':
        agent = create_structured_chat_agent(llm, tools, prompt)

    agent = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True, streaming_callback = streaming_callback, early_stopping_method='generate', max_execution_time=15
    )

    #agent_executor = AgentExecutor(agent = agent, tools=tools, verbose=False, return_intermediate_steps=False, streaming_callback = streaming_callback, handle_parsing_errors=True)

    return agent.invoke({"input": input_text})
