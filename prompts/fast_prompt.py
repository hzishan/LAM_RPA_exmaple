from langchain_core.prompts import ChatPromptTemplate

tool_use_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Make sure to use tool properly for quesetion."),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}"),
    ]
)
