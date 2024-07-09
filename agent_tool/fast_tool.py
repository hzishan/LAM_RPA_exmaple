from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import List

@tool
def click_target_tool(obj: str="送出") -> str:
    """Click the target obj in the page."""
    return "click " + obj

@tool
def type_text_tool(obj: str="申請人", text: str="王小明"):
    """Typing text into the target clicked by click_taget_tool."""
    return "type " + text + " into the target " + obj + " clicked by click_taget_tool"

page_input = ["申請人","單位", "理由", "日期", "送出"]
class TargetInput(BaseModel):
    targets: List[str] = Field(description="number of targets in page")

@tool("get_page_info", args_schema=TargetInput)
def get_page_info(targets: List[str]=page_input) -> List[str]:
    """Find/Regonition relative information in page."""
    return targets
# def regonition(targets: List[str]=page_input) -> List[str]:
#     """Regonition target image."""
#     return targets

# get_page_info = StructuredTool.from_function(
#     func= regonition,
#     name="get_page_info_tool",
#     description="Find/Regonition relative information in page.",
#     args_schema=TargetInput,
#     return_direct=True,
# )

if __name__ == "__main__":
    # print(get_page_info.invoke({"targets":page_input}))
    # print(click_target_tool.invoke("申請人"))
    print(type_text_tool.invoke({"obj":"申請人","text":"王小明"}))
