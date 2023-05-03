## Bring in deps
import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

openai_api_key = st.secrets["openai_apikey"]
google_api_key = st.secrets["google_api_key"]
google_cse_id = st.secrets["google_cse_id"]

## App Framework
st.title("Omneky Brief Bot")
with st.form('Omneky Brief Bot'):
    description = st.text_input('Plug In Your Product/Service Description Here')
    brand_name = st.text_input('Plug In Your Brand Name Here')
    value_props = st.text_input('Plug In Your Value Props Here')
    goal = st.text_input('Share Your Advertising Goal Here')
    submit_button = st.form_submit_button('Generate')

prompt = f"{description}"

## Prompt templates
title_template =  PromptTemplate(
        input_variables = ['topic'],
        template = "Write me a creative brief title about {topic}")

brief_template =  PromptTemplate(
            input_variables = ['title', 'google_research', 'value_props', 'goal', 'brand_name', 'description'],
            template = "I want you to act as a creative director. You will create a creative brief to promote a product or service of your choice. You will choose a target audience, a reason to believe for the audience, USP, and develop 10 ad concepts. Each concept should have a tagline, visual hooks, and call to action in bullet format. Make 2 concepts emotional appeals, 2 straightforward product sales, 1 concept a meme, 1 abstract concept, 2 purpose-driven concepts, 1 anti humor approach,  and 1 really random approach. Also write 10 headline ideas for digital ads. My first suggestion request is: I need help creating a creative brief for {brand_name}. The title of the brief is {title}. Their product/service description is: {description}. The brand's value props include: {value_props}. They are looking for the ads to drive {goal}. Use this google research when writing the brief: {google_research}")

## Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
brief_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


## LLMS
llm = OpenAI(temperature=0.9, max_tokens = 1000)
tools = load_tools(["google-search"], llm=llm, google_api_key = google_api_key, google_cse_id = google_cse_id)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

title_chain = LLMChain(llm=llm, prompt = title_template, verbose = True, output_key='title', memory=title_memory)
brief_chain = LLMChain(llm=llm, prompt = brief_template, verbose = True, output_key='brief', memory=brief_memory)

search = GoogleSearchAPIWrapper(google_api_key = google_api_key, google_cse_id = google_cse_id)

## show stuff to screen if there is a prompt
if submit_button:
    title = title_chain.run(prompt)
    st.write(title)
    with st.spinner('Writing Your Brief...'):
        google_research = agent.run(title)
        brief = brief_chain.run(title=title,  google_research=google_research, goal=goal, brand_name=brand_name, value_props=value_props, description=description)
    
        st.write(brief)

        with st.expander('Title History'):
            st.info(title_memory.buffer)
    
        with st.expander('Brief History'):
            st.info(brief_memory.buffer)

        with st.expander('Google Research'):
            st.info(google_research)
