from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getcwd())

from model_setting import get_llm
from langchain import hub
from agent_tool.fast_tool import click_target_tool, type_text_tool, get_page_info
from langchain.agents import AgentExecutor, create_react_agent
from prompts.chinese_prompt import fast_agent_prompt


llm = get_llm()
# prompt = hub.pull("hwchase17/react")
prompt = fast_agent_prompt

tools = [click_target_tool, type_text_tool, get_page_info]

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools,     
    handle_parsing_errors=True,
    verbose=True
)

question = "I want to click the button next to \"病假\" "
agent_executor.invoke({"input": question})

 