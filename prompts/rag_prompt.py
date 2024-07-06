from langchain.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# direct using RetriverQA chain_type_kwargs={"prompt": RAG_prompt_v1}
RAG_prompt_v1 = PromptTemplate(
    input_variables=["input"],
    template="""
    {context}: 根據文件的舉例，幫我依照格式拆解並回答問題。
    Instruction: 幫我用中文回答以下問題，再找不到答案，請回答：“我無法回答這個問題”。
    Question: {input}
    Answer: 
    """
)

RAG_prompt_v2 = PromptTemplate(
    input_variables=["question"],
    template="""
    Instruction: You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    Question: {question}
    Answer: {context}
    """
)


# llm_chain = LLMChain(llm=llm, prompt=RAG_prompt_taide, verbose=True)
RAG_prompt_taide = PromptTemplate(
    input_variables=["question"],
    template="""
    <s>
    [INST]
        <<SYS>>\n請根據文件回答問題\n<</SYS>>
        \n\n {question}
    [/INST]
    </s>"""
)

taide_template = """
    <s>[INST] <<SYS>>\n根據文件回答問題, 如果找不到請試著推測\n<</SYS>>\n\n{question} [/INST]
"""

taide_react_template = """
    <s>[INST]<<SYS>>\n 根據{tools}查詢文件並回答問題 \n<</SYS>>
    思考過程舉例: 
    Question: "如果我想交換班次，應該怎麼辦？"
    Thought: 思考該問題需要哪些步驟, 如果在搜尋的資訊中找不到答案, 則根據自己的經驗推測或是詢問是否有相關功能可以使用
    Actions 被分解成以下步驟:
    1. Action1: 使用 "Message Board" 與同事討論是否能交換班次
    2. Action2: 使用 "LeaveRequest" 申請交換班次
    3. Action3: 等待主管批准, 確認是否有相關功能可以收到通知
    4. Action4: 使用 "Personal Schedule View" 確認交換班次是否成功
    Obersevation: 觀察每個步驟的結果
    ...(Thought/Action/Observation 可以重複 N 次）
    
    輸出格式如下:
    Question: {question}
    Actions 步驟如下:
    1) Action1: ...
    2) Action2: ...
    ...
    以上是大概的步驟流程

    [/INST]</s>
"""

taide_prompt = PromptTemplate(
    input_variables=['question','tools'],
    template=taide_react_template
)
