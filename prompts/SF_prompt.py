slow_prompt = """
    {role description}
    It is difficult to code all actions in this system. We only want to code as many sub-actions as possible.The task of you is to tell me which sub-actions can be coded by you with Python.

    At each round of conversation, I will give you
    Task: T
    Context: ...
    Critique: The results of the generated codes in the last round
    
    Here are some actions coded by humans:
    {programs}
    
    You should then respond to me with
    Explain (if applicable): Why these actions can be coded by python? Are there any actions difficult to code?
    Actions can be coded: List all actions that can be coded by you.
    
    Important Tips:
    {planning tips}
    
    You should only respond in the format as described below:
    Explain: ...
    Actions can be coded:
    1) Action1: ...
    2) Action2: ...
    3) ...
"""

fast_prompt = """
    {role description}

    Here are some basic actions coded by humans:
    {programs template}
    
    Please inherit the class CodeAgent. You are only required to
    overwrite the function main function.
    
    Here are some reference examples written by me:
    {programs example}
    
    Here are the attributes of the obs that can be used:
    {obs info}
    
    Here are the guidelines of the act variable:
    {act info}
    
    At each round of conversation, I will give you
    Task: ...
    Context: ...
    Code from the last round: ...
    Execution error: ...
    Critique: ...
    
    You should then respond to me with
    Explain (if applicable): Can the code complete the given action? What does the chat log and execution error imply?
    
    You should only respond in the format as described below:
    {code format}
"""