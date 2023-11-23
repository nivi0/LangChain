from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI
import langchain_helper
from langchain.utilities import WikipediaAPIWrapper
from secret_key import serpapi_key

import os 
os.environ['SERPAPI_API_KEY'] = serpapi_key

wikipedia = WikipediaAPIWrapper()

tools = load_tools(["serpapi","wikipedia", "llm-math"], llm=langchain_helper.llm)

agent = initialize_agent(
    tools,
    langchain_helper.llm,
    agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
if __name__ == "__main__":
    print(agent.run("When was Elon Musk born? What is his age right now in Nov 2023?"))
    # print(agent.run("What was the GDP of US in 2023 plus 5?"))