def _deprecated_RetrievalQA(llm, doc_name="ScheduleDesign.docx"):
    from prompts.rag_prompt import RAG_prompt_taide
    from langchain.chains.retrieval_qa.base import RetrievalQA
    from langchain.chains.llm import LLMChain
    from agent_tool.Indexing import indexing

    retriver = indexing(doc_name=doc_name)

    LLMChain(llm=llm, prompt=RAG_prompt_taide, verbose=True)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriver,
        chain_type="stuff",
        # chain_type_kwargs={"prompt": RAG_prompt}, 
        verbose=True,
    )
    return qa

def legacy_RAG(llm, prompt, separators=["。\n\n"], doc_name="ScheduleDesign.docx", **kwargs):
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain
    from agent_tool.Indexing import indexing

    retriever = indexing(doc_name, separators, **kwargs)
    chain_with_prompt = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, chain_with_prompt)
    # rag_chain.invoke({"input": query})

    return rag_chain


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def LCEL_RAG(llm, prompt, doc_name="ScheduleDesign.docx", **kwargs):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from agent_tool.Indexing import indexing

    retriever = indexing(doc_name=doc_name, **kwargs)

    qa_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
    )
    return qa_chain


    
from pathlib import Path
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getcwd())
    from model_setting import get_llm


    # chain = legacy_RAG(get_llm(model_name="taide"), db_name="SystemDesign_taideVDB")
    # result = chain.invoke({"input":"我想請假應該使用什麼功能呢"})
    # print(result)

    llm = get_llm(model_name="breeze")
    # chain = legacy_RAG(llm=llm, db_name="SystemDesign_breezeVDB")
    # result = chain.invoke({"input":"我想請假應該使用什麼功能呢"})
    # print(result)
    from prompts.rag_prompt import rag_prompt_v2
    chain = LCEL_RAG(llm=llm, prompt=rag_prompt_v2, doc_name="ScheduleDesign.docx", db_name="SystemDesign_breezeVDB")
    result = chain.invoke("我想請假應該使用什麼功能呢")
    # result = chain.invoke("這個系統有甚麼功能?")
    # print(result)



    
