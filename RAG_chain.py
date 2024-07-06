from langchain_community.document_loaders import PyMuPDFLoader,TextLoader,Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from os.path import dirname
import os

from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)

################ model setting ################
emodel = "GanymedeNil/text2vec-large-chinese"
emodel_name = "text2vec-large-chinese"

model_list = ["Breeze-7B-Instruct-v0.1-Q8_0","TAIDE-LX-7B-Chat.Q8_0"]
model = model_list[0]

################ Indexing ################
# Load
QList = ["我想請假","我需要寫工作日誌，可以幫我生一個草稿嗎","我想看這個月的班表"]
# QList = ["排班系統有什麼特別的工具嗎","我需要登入才能使用這個系統嗎","我臨時有事需要調班，應該怎麼做"]
loader = Docx2txtLoader("documents/ScheduleDesign.docx")
data = loader.load()


# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=5, separators=["。\n"])
all_splits = text_splitter.split_documents(data)  

# Embed and store (embedding model)
emodel_path = (dirname(__file__)) + "\emodel\\" + emodel_name
if not os.path.exists(emodel_path):
    print("model not found in", emodel_path)
else:
    embedding_model = HuggingFaceEmbeddings(model_name=emodel_path, model_kwargs={'device': 'cpu'})

db_path = 'db-SD-'+ emodel_name
if not os.path.exists(db_path):
    # save to disk
    vectorDB = Chroma.from_documents(documents=all_splits, embedding=embedding_model, persist_directory=db_path, collection_metadata={"hnsw:space": "cosine"})
else: 
    # load from disk 
    vectorDB = Chroma(persist_directory=db_path, embedding_function=embedding_model)


# Testing embedding_model with documents

# print("============== similar search ===============")
# results = vectorDB.similarity_search_with_score(QList[0])
# for result, score in results:
#     print(result.page_content)
#     print("cosine similarity: ",score)

# for i in range(len(QList)):
#     result, score = vectordb.similarity_search_with_score(QList[i])[0] 
#     print("============== similar search ===============")
#     print(QList[i])
#     print(result.page_content)
#     print("cosine similarity: ",score)

################ loading LLM model ################
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain_community.llms import LlamaCpp
from langchain.chains.llm import LLMChain

model_path = (dirname(__file__)) + "/model/" + model +'.gguf'
if not os.path.exists(model_path):
    print("model not found in", model_path)
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

# print("=========== model_result ================")
# for i in range(len(QList)):
#     llm.invoke(QList[i])
#     print("=====================================")


################ define prompt template ################
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# RAG_prompt = PromptTemplate(
#     input_variables=["question"],
#     template="""
#     {context}: 根據文件的舉例，幫我依照格式拆解並回答問題。
#     # Instruction: 幫我用中文回答以下問題，如果在文件中找不到相關信息，可以試著使用 google 搜尋功能，再找不到答案，請回答：“我無法回答這個問題”。
#     Question: {question}
#     Answer: 
#     """
# )

RAG_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
    <s>
    [INST]
        <<SYS>>\n你是台灣某公司主管，請根據文件回答問題\n<</SYS>>
        \n\n {question}
    [/INST]
    </s>"""
)

################ augmentation: query + retrieved information ################
llm_chain = LLMChain(llm=llm, prompt=RAG_prompt, verbose=True)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorDB.as_retriever(include_metadata=False),
    chain_type="stuff",
    # chain_type_kwargs={"prompt": RAG_prompt}, 
    verbose=True,
)

# print("=========== RAG_result ================")
for i in range(len(QList)):
    print(qa.invoke(QList[i]))
    print("=====================================")
