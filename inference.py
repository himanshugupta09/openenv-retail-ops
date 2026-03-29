import os
import json
from openai import OpenAI
from env import RetailOpsEnv
from models import Action

def run_inference():
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        raise ValueError("CRITICAL: HF_TOKEN environment variable is missing!")

    client = OpenAI(
        base_url=os.environ.get("API_BASE_URL"),
        api_key=api_key
    )
    model_name = os.environ.get("MODEL_NAME")
    
    env = RetailOpsEnv()
    tasks = ["easy_escalation", "medium_refund", "hard_reconciliation"]

    for task in tasks:
        print(f"\n--- Starting Task: {task} ---")
        obs = env.reset(task_id=task)
        done = False
        
        while not done and env.step_count < 10:
            prompt = f"""
            You are an e-commerce backend agent.
            Current Ticket: {obs.ticket_text}
            System Message: {obs.system_message}
            Last DB Query Result: {obs.db_result}
            
            You must choose an action. 
            Available tools and their required args:
            - query_database: {{"order_id": "string"}}
            - issue_refund: {{"order_id": "string"}}
            - restock_inventory: {{"item": "string"}} (You must use the actual item name from the DB query, e.g., 'wrong_item')
            - escalate_ticket: {{}}
            - submit_final_answer: {{}}
            
            Respond ONLY with a valid JSON object. 
            Example format:
            {{"tool_name": "query_database", "tool_args": {{"order_id": "456"}}}}
            """
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={ "type": "json_object" }
                )
                
                action_data = json.loads(response.choices[0].message.content)
                action = Action(**action_data)
                
                print(f"Agent Action: {action.tool_name} | Args: {action.tool_args}")
                obs, reward, done, info = env.step(action)
                
            except Exception as e:
                print(f"Agent produced invalid format or API error: {e}")
                obs, reward, done, info = env.step(Action(tool_name="submit_final_answer", tool_args={}))

        print(f"Task {task} finished. Final Score: {info['score']}/1.0")

if __name__ == "__main__":
    run_inference()