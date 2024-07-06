from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)


def main():
    from model_setting import get_llm
    from agent_tool.fast_tool import click_target_tool, type_text_tool, get_page_info
    from agent_chain.retrieverQA_chain import create_vectordb, indexing, rag_chain

    # rag_llm = get_llm(model_name="breeze")
    # rag_chain(rag_llm, query="我想請假應該使用什麼功能呢")
    
    required_params = {
        "必填": ["申請人", "單位", "請假類型", "開始日期", "結束日期"],
        "可選": ["代班人員", "請假事由", "備註"],
    }

    
    indexing(doc_name="leave_info.txt")
    # step2 = rag_chain(get_llm(model_name="breeze"), doc_name="leave_info.txt")
    # step2.invoke({"input":"how many variables need to fill?"})
    

    # agent = create_react_agent(llm, tools, prompt)

    # agent_executor = AgentExecutor(
    #     agent=llm, 
    #     tools=tools,     
    #     agent_executor_kwargs={"call_tools": call_tools},
    #     handle_parsing_errors=True,
    #     verbose=True
    # )


    # question = "填好請假表"
    # question = "點擊申請人旁邊的按鈕，後輸入王小明"
    # question = "輸入王小明"
    # agent_executor.invoke({"input": question})


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getcwd())
    main()