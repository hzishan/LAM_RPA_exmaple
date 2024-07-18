from pathlib import Path

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
        vectorDB = Chroma(persist_directory=db_path, embedding_function=embedding_model)
        return vectorDB
    except Exception as e:
        print(f"Error: {e}")
        return None

def indexing(doc_name="ScheduleDesign.docx", separators=["。\n\n"], **kwargs):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from model_setting import get_emodel
    
    data = load_document(doc_name)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128,
        chunk_overlap=10,
        separators = separators
    )
    docs_splits = text_splitter.split_documents(data)

    embedding_model = get_emodel()

    try: 
        db_name = kwargs["db_name"]
    except KeyError:
        db_name = doc_name.split(".")[0]+"_VectorDB"

    vectorDB = create_vectordb(docs_splits, embedding_model, db_name)
    retriever = vectorDB.as_retriever(**kwargs)
    return retriever  

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getcwd())

    retriever = indexing(
        doc_name="ScheduleDesign.docx",
        separators=["。\n\n"],
        db_name="testing_db",
        search_type="similarity",
        search_kwargs={"k": 3}
        )
    
    retriever2 = indexing(
        doc_name="ScheduleDesign.docx",
        separators=["。\n\n"],
        db_name="testing_db",
        search_type="mmr",
        search_kwargs={"k":3}
    )

    retriever3 = indexing( # 跟retriever基本上一樣
        doc_name="ScheduleDesign.docx",
        separators=["。\n\n"],
        db_name="testing_db",
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.6}
    )

    for i in [retriever,retriever2]:
        result = i.invoke("我想請假應該使用什麼功能呢")
        print(f"============{i}==================")
        for j in result:
            print(j)
