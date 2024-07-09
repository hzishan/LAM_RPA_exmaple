from langchain_core.prompts import ChatPromptTemplate

router_template ="""
    Given the user task below, classify it as either being about `click_target_tool`, `type_text_tool`, or `get_page_info`.
    Do not respond with more than one word.

    <question>
    {task}
    </question>
    
    Classification:
"""

order_template = """

"""

tool_route_template = """
    Given the user task below, classify it as one of the following tools: {tools}.
    And return only tool_name(variables) in the response.
    For example: type_text_tool("申請人", "王小明")
    <question>
    {task}
    </question>
    
    Classification:
"""
