from models import Observation, Action
from tasks import TASKS, grade_task
import copy

class RetailOpsEnv:
    def __init__(self):
        self.current_task_id = None
        self.db = {}
        self.ticket = ""
        self.action_history = []
        self.step_count = 0

    def reset(self, task_id: str = "easy_escalation") -> Observation:
        self.current_task_id = task_id
        task_data = TASKS[task_id]
        self.ticket = task_data["ticket"]
        self.db = copy.deepcopy(task_data["initial_db"])
        self.action_history = []
        self.step_count = 0
        
        return Observation(
            ticket_text=self.ticket,
            system_message="Environment reset. Awaiting action."
        )

    def state(self) -> dict:
        return {
            "task_id": self.current_task_id,
            "db": self.db,
            "action_history": self.action_history,
            "steps": self.step_count
        }

    def step(self, action: Action):
        self.step_count += 1
        self.action_history.append(action.model_dump())
        
        reward = 0.0
        done = False
        obs = Observation(ticket_text=self.ticket, system_message="")

        tool = action.tool_name
        args = action.tool_args

        try:
            if tool == "query_database":
                order_id = str(args.get("order_id", ""))
                obs.db_result = {order_id: self.db.get(order_id, "Not found")}
                obs.system_message = "Database queried successfully."
                reward += 0.1 

            elif tool == "issue_refund":
                order_id = str(args.get("order_id", ""))
                if order_id in self.db and not self.db[order_id].get("refunded"):
                    self.db[order_id]["refunded"] = True
                    obs.system_message = f"Refund issued for {order_id}."
                    reward += 0.2
                else:
                    obs.system_message = "Invalid refund request."
                    reward -= 0.1 

            elif tool == "restock_inventory":
                item = str(args.get("item", ""))
                if "inventory" in self.db and item in self.db["inventory"]:
                    self.db["inventory"][item] += 1
                    obs.system_message = f"Restocked {item}."
                    reward += 0.2
                else:
                    obs.system_message = "Item not found in inventory."
                    reward -= 0.1

            elif tool in ["escalate_ticket", "submit_final_answer"]:
                obs.system_message = "Task completed by agent."
                done = True # This correctly stops the loop

        except Exception as e:
            obs.system_message = f"Action failed: {str(e)}"
            reward -= 0.2

        if done or self.step_count >= 10:
            done = True
            final_score = grade_task(self.current_task_id, self.state(), self.action_history)
            if final_score >= 0.8:
                reward += 1.0
            elif final_score == 0.0:
                reward -= 0.5

        info = {"score": grade_task(self.current_task_id, self.state(), self.action_history)}
        return obs, reward, done, info