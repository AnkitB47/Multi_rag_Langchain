from typing import Optional, TypedDict

class GraphState(TypedDict):
    input: str
    lang: str
    pdf_path: Optional[str]
    image_path: Optional[str]
    search_result: Optional[str]
    pdf_result: Optional[str]
    image_result: Optional[str]
    final_output: Optional[str]
