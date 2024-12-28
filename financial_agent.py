from phi.agent import Agent
#from phi.model.yfinance import YahooFinanceModel
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai

import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

## web search Agent
web_search_agent=Agent(
    name='web_search_agent',
    role="search the web for information",
    model=Groq(id="llama-3.2-1b-preview"),
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

multi_ai_agent=Agent(
    team=[web_search_agent, financial_agent],
    instructions=["Always include sources","Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for Coforge Ltd. (NSE:COFORGE)",
                              stream=True)