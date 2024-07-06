def load_document(doc_name):
    from langchain_community.document_loaders import PyMuPDFLoader,TextLoader,Docx2txtLoader
    doc_path = Path.cwd() / "documents" / doc_name
    case = doc_name.split(".")[-1]
    if case == "pdf":
        loader = PyMuPDFLoader(str(doc_path))
    elif case == "txt":
        loader = TextLoader(str(doc_path), encoding="utf-8")
    elif case == "docx":
        loader = Docx2txtLoader(str(doc_path))
    else:
        print("File type not supported.")
        return None
    return loader.load()

def create_vectordb(document, embedding_model, db_name = "VectorDB"):
    from langchain_community.vectorstores import Chroma
    db_path = str(Path.cwd() / db_name)
    try:
        vectorDB = Chroma.from_documents(
            documents=document,
            embedding=embedding_model,
            persist_directory=db_path,
            collection_metadata={"hnsw:space": "cosine"}
        )
        return vectorDB
    except FileNotFoundError:
        vectorDB = Chroma(persist_directory=db_path, embedding_model=embedding_model)
        return vectorDB
    except Exception as e:
        print(f"Error: {e}")
        return None

def indexing(doc_name="ScheduleDesign.docx", knum=3, sep_list=["。\n","\nQ"], **kwarg):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from model_setting import get_emodel
    
    data = load_document(doc_name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=5,
        separators=sep_list
    )
    all_splits = text_splitter.split_documents(data)

    embedding_model = get_emodel()

    try: 
        db_name = kwarg["db_name"]
    except KeyError:
        db_name = doc_name.split(".")[0]+"_VectorDB"

    vectorDB = create_vectordb(all_splits, embedding_model, db_name)
    retriever = vectorDB.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": knum},
                    include_metadata=False)
    return retriever

def _deprecated_rag_LCEL(llm, query="我想請假應該使用什麼功能呢", doc_name="ScheduleDesign.docx"):
    from prompts.rag_prompt import RAG_prompt_taide
    from langchain.chains.retrieval_qa.base import RetrievalQA
    from langchain.chains.llm import LLMChain
    # from langchain.chain improt RetrievalQA, LLMChain
    
    retriver = indexing(doc_name=doc_name)

    LLMChain(llm=llm, prompt=RAG_prompt_taide, verbose=True)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriver,
        chain_type="stuff",
        # chain_type_kwargs={"prompt": RAG_prompt}, 
        verbose=True,
    )
    return qa.invoke(query)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def new_retrievalQA(llm, doc_name="ScheduleDesign.docx"):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from prompts.rag_prompt import RAG_prompt_v1, RAG_prompt_v2

    retriever = indexing(doc_name=doc_name)
    prompt = RAG_prompt_v2

    qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
    )
    return qa_chain


def rag_chat_chain(llm, doc_name="ScheduleDesign.docx"):
    from prompts.rag_prompt import RAG_prompt_v1, RAG_prompt_v2
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains.retrieval import create_retrieval_chain

    retriever = indexing(doc_name=doc_name)
    chain_with_prompt = create_stuff_documents_chain(llm, RAG_prompt_v1)
    rag_chain = create_retrieval_chain(retriever, chain_with_prompt)
    # rag_chain.invoke({"input": query})

    return rag_chain


    
from pathlib import Path
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getcwd())

    from model_setting import get_llm
    # _deprecated_rag_LCEL(get_llm(model_name="breeze"))
    # rag_chain(get_llm(model_name="breeze"), query="我想請假應該使用什麼功能呢")

    # chain = rag_chain(get_llm(model_name="taide"))
    # chain.invoke({"input":"我想請假應該使用什麼功能呢"})

    chain = new_retrievalQA(get_llm(model_name="breeze"))
    chain.invoke("我想請假應該使用什麼功能呢")




    
