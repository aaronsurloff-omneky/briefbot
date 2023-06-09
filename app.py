## Bring in deps
import os
import requests
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from bs4 import BeautifulSoup

#api key setup
openai_api_key = st.secrets["openai_apikey"]
google_api_key = st.secrets["google_api_key"]
google_cse_id = st.secrets["google_cse_id"]

## Goal Options
Fundraising = "Fundraising"
Installs = "Installs"
Leads = "Leads"
Purchases = "Purchases"
Subscribers = "Subscribers"
Donations = "Donations"

## App Framework
st.title("Omneky Brief Bot")
with st.form('Omneky Brief Bot'):
    brand_name = st.text_input('Plug In Your Brand Name Here')
    website = st.text_input('Plug In Your Website URL Here', placeholder="https://wwww.website.com")  
    description = st.text_input('Plug In Your Product/Service Description Here')
    with st.expander('Advanced Inputs'):
        value_props = st.text_input('Plug In Your Value Props Here', placeholder="Ex. saves time and money")
        goal = st.selectbox(
            label="Advertising Goal",
            options=[Fundraising, Installs, Leads, Purchases, Subscribers, Donations])
        purpose = st.text_input('Plug In Your Product/Service Purpose Here', placeholder="Ex. reduce carbon emissions")
    submit_button = st.form_submit_button('Generate')

prompt = f"{description}"

## Prompt templates
title_template =  PromptTemplate(
        input_variables = ['topic'],
        template = "Write me a creative brief title about {topic}")

brief_template =  PromptTemplate(
            input_variables = ['title', 'google_research', 'value_props', 'goal', 'brand_name', 'description', 'text_summary', 'purpose'],
            template = "I want you to act as a creative director. You will create a creative brief to promote a product or service of your choice. You will choose a target audience, a reason to believe for the audience, USP, and develop 7 ad concepts. Each concept should have a tagline, visual hooks, and call to action in bullet format. Make 1 concepts emotional appeals, 1 straightforward product sales, 1 concept a meme, 1 abstract concept, 1 purpose-driven concepts, 1 anti humor approach,  and 1 really random approach. Also write 1 headline idea for digital ads. My first suggestion request is: I need help creating a creative brief for {brand_name}. The title of the brief is {title}. Their product/service description is: {description}. The brand's value props include: {value_props}. They brand purpose is: {purpose}. They are looking for the ads to drive {goal} generation. Use this google research on good ad concepts writing the brief: {google_research}. Also use the follow research of {brand_name}'s website: {text_summary}")

website_template =  PromptTemplate(
        input_variables = ['brand_name','first_1000_chars'],
        template = "Explain the product/service description for {brand_name} based on this scape of their website in 3 sentences: {first_1000_chars}")

## Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
brief_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


## LLMS
llm = OpenAI(temperature=0.9, max_tokens = 3000, openai_api_key = openai_api_key)
tools = load_tools(["google-search"], llm=llm, google_api_key = google_api_key, google_cse_id = google_cse_id)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

title_chain = LLMChain(llm=llm, prompt = title_template, verbose = True, output_key='title', memory=title_memory)
brief_chain = LLMChain(llm=llm, prompt = brief_template, verbose = True, output_key='brief', memory=brief_memory)
website_chain = LLMChain(llm=llm, prompt = website_template, verbose = True, output_key='website_summary')

##search setup
search_prompt = f"top creative ideas for {goal} ads"
search = GoogleSearchAPIWrapper(google_api_key = google_api_key, google_cse_id = google_cse_id)

## show stuff to screen if there is a prompt
if submit_button:
    with st.spinner('Writing Your Brief Title...'):
        title = title_chain.run(prompt)
        st.write(title)
    with st.spinner('Researching Your Website...'):
        web_response = requests.get(website)
        html_content = web_response.text
        soup = BeautifulSoup(html_content, "html.parser")
        rawtext = soup.get_text()
    
    ## Extra first 1000 characters to stay within token limit
        first_1000_chars = rawtext[:1000]
    
    ## Create summary of website research
        text_summary = website_chain.run(brand_name=brand_name, rawtext=rawtext, first_1000_chars=first_1000_chars)
    
    ## Research Google
    with st.spinner('Researching Google...'):
        google_research = agent.run(search_prompt)
    
     ## Write Creative Brief
    with st.spinner('Writing Your Brief...'):
        google_research = agent.run(search_prompt)
        brief = brief_chain.run(title=title,  google_research=google_research, goal=goal, brand_name=brand_name, value_props=value_props, description=description, text_summary=text_summary)
    
        st.write(brief)
        ## Display the memory of each action taken
        with st.expander('Title History'):
            st.info(title_memory.buffer)
        with st.expander('Blog History'):
            st.info(brief_memory.buffer)
        with st.expander('Google Research'):
            st.info(google_research)        
        with st.expander('Website Research'):
            st.info(text_summary)       
        with st.expander('Website Research Raw'):
            st.info(rawtext)
