import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
from phi.model.groq import Groq
import os
import phi
from phi.playground import Playground,serve_playground_app
from phi.utils import *
from phi.api import *
#load environment variables from .env file
load_dotenv()

# agent = Agent(
#     # Note: 'llm' is now 'model' and 'model' is now 'id'
#     model=OpenAIChat(id="gpt-4o"),
# )

phi.api=os.getenv("PHI_API_KEY")
#openai.api_key = os.getenv("OPENAI_API_KEY")

## web search Agent
web_search_agent=Agent(
    name='web_search_agent',
    role="search the web for information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

##financial Agent
financial_agent=Agent(
    name="Finance AI Agent",
    #role="search the web for financial information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, 
                        stock_fundamentals=True,
                        company_news=True
                        ),],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

# multi_ai_agent=Agent(
#     team=[web_search_agent, financial_agent],
#     instructions=["Always include sources","Use tables to display the data"],
#     show_tool_calls=True,
#     markdown=True
# )

app=Playground(agents=[financial_agent, web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app", reload=True)
