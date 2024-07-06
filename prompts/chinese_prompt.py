from langchain.prompts import PromptTemplate


react_template = """
    將問題切割成數個且符合特殊條件的步驟, 並從 agent 挑選最合適的工具: {tools}, 回答問題.
    
    特殊條件: 
    {tool_names} 不可將其翻譯
    被 click 的物件, 必須先被 rag_image 辨識出來過
    type_text_tool 執行前, 必須先 click_button_tool
    type_text_tool 不一定會被呼叫

    使用範例格式:
    Question: 輸入2020-11-03
    Action1: 使用 reg_image 辨識出日期
    Action2: 使用 click_button_tool 點擊日期
    Action3: 使用 type_text_tool 輸入 "2020-11-03"
    ... ( ActionN 可以重複 N 次)
    Final Answer: 完成任務

    從以下開始!
    Question: {input}
    Action1: {agent_scratchpad}
"""

fast_template = """
    將指示/動作切割成數個步驟, 且需符合特殊條件, 並從 agent 挑選合適的工具: {tools} 完成子步驟.
    
    特殊條件: 
    {tool_names}不能被翻譯, 其餘用繁體中文回覆
    如果需要呼叫 get_page_info 則在 click_target_tool 前執行
    type_text_tool 執行前, 必須先 click_target_tool
    type_text_tool 不一定會被呼叫

    以下是參考範例:
    Task: input 要解決的任務
    Thought: 思考該任務需要什麼動作，被分解的多細部
    Action: 根據 {tool_names} 選擇一個動作
    Action Input: 動作需要的輸入目標或文字
    Observation: 動作的結果
    ... ( Thought / ActionN / Action Input / Observation 可以重複 N 次)
    Final Answer: 完成任務

    從以下開始!輸出結果
    Task: {input}
    Thought: {agent_scratchpad}
"""

fast_taide_template = """
    <s>[INST]
    <<SYS>> 根據使用者的input完成任務, 可拆分成多個子任務, 從 {tools} 中選擇合適的工具完成.
        下面是一些完成任務要符合的條件: 
            除了 {tool_names}, 其餘用繁體中文回覆
            點擊過的物件, 才能開始輸入文字
            不是所有物件都需要被輸入

        輸出參考格式:
            Task: 要解決的任務
            Thought: 思考該任務首先需要什麼動作
            Action: 根據 {tool_names} 選擇一個動作
            Action Input: 動作需要的輸入目標或文字
            Observation: 動作的結果, 是否需要繼續下一步
            ... ( Thought / ActionN / Action Input / Observation 可以重複 N 次)
            Final Answer: 完成任務

        以下是一個範例:
            Task: 輸入2020-11-03
            Thought: 需要先點擊可輸入的"日期"的相關欄位
            Action: 根據 {tool_names} 選擇, click_target_tool
            Action Input: click_target_tool.invoke("日期")
            Observation: 點擊日期欄位
            Thought: 接著需要輸入日期 "2020-11-03"
            Action: 根據 {tool_names} 選擇, type_text_tool
            Action Input: type_text_tool.invoke("2020-11-03")
            Observation: 輸入日期 "2020-11-03"
            Final Answer: 完成任務
    <</SYS>>
    Task: {input}
    Thought: {agent_scratchpad}
    [/INST]
"""

fast_agent_prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
    # template=fast_template
    template = fast_template
)