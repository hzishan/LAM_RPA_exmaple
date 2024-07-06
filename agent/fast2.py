import asyncio

async def main():
    from model_setting import get_llm
    from langchain import hub
    from agent_tool.fast_tool import click_target_tool, type_text_tool, get_page_info
    from langchain.agents import AgentExecutor, create_react_agent
    from prompts.chinese_prompt import fast_agent_prompt

    prompt = hub.pull("hwchase17/react")
    # prompt = fast_agent_prompt

    tools = [click_target_tool, type_text_tool, get_page_info]

    llm = get_llm(model_name="breeze")

    agent = create_react_agent(llm, tools, prompt)

    question = "I want to click the button next to \"申請人\" "

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools,
        handle_parsing_errors=True,
        verbose=True
    )

    # agent_executor.invoke({"input": question})
    for step in agent_executor.iter({"input": question}):
        if output := step.get("intermediate_steps"):
            action, value = output[0]
            if action == "tool":
                tool_name = value
                tool = next((t for t in tools if t.name == tool_name), None)
                if tool:
                    print(f"Calling tool: {tool.name}")
                    tool.use(value)
                else:
                    print(f"Invalid tool: {tool_name}")
            elif action == "observation":
                print(f"Observation: {value}")
            elif action == "thought":
                print(f"Thought: {value}")
        elif "output" in step:
            print(f"Final Answer: {step['output']}")


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getcwd())
    asyncio.run(main())