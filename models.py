from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, Optional

class Observation(BaseModel):
    ticket_text: str
    db_result: Optional[Dict[str, Any]] = None
    system_message: str

class Action(BaseModel):
    tool_name: Literal[
        "query_database", 
        "issue_refund", 
        "restock_inventory", 
        "escalate_ticket", 
        "submit_final_answer"
    ]
    tool_args: Dict[str, Any] = Field(default_factory=dict)