from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)

def route(tools_list, tool_name, **kwargs):
    for tool in tools_list:
        print("tool", tool.name)
        if tool_name in tool.name:
            return tool.invoke({**kwargs})
    return f"Do not find tool named {tool_name} "

def classify_chain(tools):
    from model_setting import get_llm
    from agent_tool.fast_tool import click_target_tool, type_text_tool, get_page_info
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from prompts.fast_prompt import router_template, tool_route_template
    chain = (
        PromptTemplate.from_template(
            # router_template
            tool_route_template
        )
        | get_llm(model_name="breeze")
        | StrOutputParser()
    )

    tools = [click_target_tool, type_text_tool, get_page_info]
    result=chain.invoke({"task": "I want to click the button next to 病假", "tools": tools})
    print(route(tools, result.strip(), obj="病假"))
    # result = chain.invoke({"task": "typing 王小明 to inputbox 申請人"})
    # print(route(tools, "type_text_tool", text="王小明", obj="申請人"))


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getcwd())
    classify_chain()