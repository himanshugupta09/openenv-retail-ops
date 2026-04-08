from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel
from models import Action

# The Observation state returned to the LLM at every step
class Observation(BaseModel):
    ticket_text: str
    db_result: Optional[Dict[str, Any]] = None
    system_message: str

class RetailOpsEnv:
    def __init__(self):
        self.step_count = 0
        self.task_id = ""
        self.state_data = {}
        # Mock database initialized at the start of the environment
        self.mock_db = {
            "456": {"status": "shipped", "item": "Laptop Stand", "refunded": False},
            "789": {"status": "shipped", "item": "Mechanical Keyboard", "refunded": False, "wrong_item_shipped": True}
        }
        self.obs = Observation(ticket_text="", system_message="")

    def reset(self, task_id: str = "easy_escalation") -> Observation:
        self.step_count = 0
        self.task_id = task_id
        
        # Reset task tracking states
        self.state_data = {
            "escalated": False, 
            "refunded": False, 
            "restocked": False
        }
        
        # Load the correct ticket scenario based on the task_id
        if task_id == "easy_escalation":
            self.obs = Observation(
                ticket_text="I DEMAND TO SPEAK TO A MANAGER IMMEDIATELY! THIS IS UNACCEPTABLE!",
                system_message="New ticket received."
            )
        elif task_id == "medium_refund":
            self.obs = Observation(
                ticket_text="My order 456 arrived damaged. Please refund me.",
                system_message="New ticket received."
            )
        elif task_id == "hard_reconciliation":
            self.obs = Observation(
                ticket_text="I ordered a mouse but received a Mechanical Keyboard for order 789. Cancel and refund it.",
                system_message="New ticket received."
            )
        else:
            self.obs = Observation(ticket_text="Unknown task.", system_message="Error")
            
        return self.obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = 0.01  # Safe minimum reward
        done = False
        
        # Tool Execution Logic
        if action.tool_name == "query_database":
            order_id = action.tool_args.get("order_id", "")
            if order_id in self.mock_db:
                self.obs.db_result = self.mock_db[order_id]
                self.obs.system_message = f"Database queried for order {order_id}."
                reward = 0.5
            else:
                self.obs.db_result = None
                self.obs.system_message = "Order not found."
                
        elif action.tool_name == "issue_refund":
            order_id = action.tool_args.get("order_id", "")
            if order_id in self.mock_db and not self.mock_db[order_id]["refunded"]:
                self.mock_db[order_id]["refunded"] = True
                self.state_data["refunded"] = True
                self.obs.system_message = f"Refund issued for order {order_id}."
                reward = 0.5
            else:
                self.obs.system_message = "Invalid refund request or already refunded."
                
        elif action.tool_name == "restock_inventory":
            item = action.tool_args.get("item", "")
            if item:
                self.state_data["restocked"] = True
                self.obs.system_message = f"Inventory restocked for item: {item}."
                reward = 0.5
            else:
                self.obs.system_message = "Invalid restock request."

        elif action.tool_name == "escalate_ticket":
            self.state_data["escalated"] = True
            self.obs.system_message = "Ticket escalated to human manager."
            done = True
            
        elif action.tool_name == "submit_final_answer":
            self.obs.system_message = "Task marked as resolved."
            done = True
            
        else:
            self.obs.system_message = f"Unknown tool: {action.tool_name}"

        # Hard limit to prevent infinite loops
        if self.step_count >= 10:
            done = True

        # --- CRITICAL SCORE CLAMPING FIX ---
        # The OpenEnv validator requires scores to be strictly between 0 and 1. 
        # We use 0.01 for failure and 0.99 for success.
        score = 0.01  
        
        if done:
            if self.task_id == "easy_escalation" and self.state_data["escalated"]:
                score = 0.99
            elif self.task_id == "medium_refund" and self.state_data["refunded"]:
                score = 0.99
            elif self.task_id == "hard_reconciliation" and self.state_data["refunded"] and self.state_data["restocked"]:
                score = 0.99

        info = {"score": score}
        return self.obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "task_id": self.task_id,
            "observation": self.obs.model_dump(),
            "db": self.mock_db
        }
