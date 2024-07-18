def module_rpa():
    from model_setting import get_llm
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain.prompts import PromptTemplate
    from agent_chain.retrieverQA_chain import LCEL_RAG, legacy_RAG
    from prompts.rag_prompt import rag_prompt_v1, rag_prompt_v2

    llm = get_llm(model_name="breeze")
    
    # retrieval_chain = legacy_RAG(llm, rag_prompt_v2, doc_name="ScheduleDesign.docx", separators=["。\n\n"])
    retrieval_chain = LCEL_RAG(llm, rag_prompt_v1 , doc_name="ScheduleDesign.docx", separators=["。\n\n"])

    translate_prompt = PromptTemplate.from_template(
        """
        You are an translator. Translate the following chinese text to english:
        Input: {rag_result}
        Result: 
        """
    )
    translate_chain = translate_prompt | llm | StrOutputParser()


    from robotiive import rpa_runner, get_task_name
    
    rpa_calling_prompt = PromptTemplate.from_template(
        """
        Pick up only one scripts from {RPA_scipts} based on the following input: 
        Input: {eng_result}
        Result:
        """
    )
    rpa_calling_chain = rpa_calling_prompt | llm | StrOutputParser()


    final_chain = ({
        "rag_result" : retrieval_chain,
        "RPA_scipts": lambda x : get_task_name("LAM")
        } 
        | RunnablePassthrough.assign(eng_result=translate_chain)
        | RunnablePassthrough.assign(RPA_scipts=rpa_calling_chain)
    )

    # result = final_chain.invoke({"input": "我想請假應該使用什麼功能呢"}) # for legacy_RAG
    result = final_chain.invoke("我想請假應該使用什麼功能呢") # for LCEL_RAG
    print(result["RPA_scipts"].strip())
    task_name = result["RPA_scipts"].strip()
    rpa_runner(task_name)


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getcwd())
    module_rpa()