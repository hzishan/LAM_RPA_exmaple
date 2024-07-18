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
    Given the user task below, classify it as either being about {tools}.
    Do not respond with more than one word.

    <question>
    {task}
    </question>
    
    Classification:
"""
