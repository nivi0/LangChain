from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import os 
from secret_key import openapi_key
os.environ['OPENAI_API_KEY'] = openapi_key

llm = OpenAI(temperature=0.7)

# Chain 1: cricket world cup year
pt_cricket_WC_year = PromptTemplate(
    input_variables=['game'],
    template="What year the last {game} world cup held?"
)

year_chain = LLMChain(llm=llm, prompt=pt_cricket_WC_year, output_key='wc_year')


# # Chain 2: Team name
# pt_cricket_team = PromptTemplate(
#     # input_variables='wc_year',
#     template="Which team won the match?"
# )

# team_chain = LLMChain(llm=llm, prompt=pt_cricket_team, output_key='team')

memory = ConversationBufferMemory() # Endless 

chain = SequentialChain(
    chains=[year_chain],
        input_variables=['game'],
        output_variables=['wc_year']
    )

response = chain({'game':'cricket'})
print(response)

memory = ConversationBufferWindowMemory(k=3)
convo = ConversationChain(
    llm=OpenAI(temperature=0.7), 
    memory=memory
    ) 

print("*******Initial Prompt Template:\n",convo.prompt.template)

print(convo.run("Which team won the {}?".format(response['wc_year'])))
print(convo.run("Who was the captian?"))
print(convo.run("India vs. England"))
print(convo.run("5 + 6 + 1"))

print("*******Convo Memory:\n",convo.memory.buffer)
