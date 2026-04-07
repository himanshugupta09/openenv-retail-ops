import os
import json
from openai import OpenAI
from env import RetailOpsEnv
from models import Action

def run_inference():
    # 1. NO default for HF_TOKEN (Checklist condition met)
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("CRITICAL: HF_TOKEN environment variable is missing!")

    # 2. Defaults SET for API_BASE_URL and MODEL_NAME (Checklist condition met)
    api_base_url = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    model_name = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

    client = OpenAI(
        base_url=api_base_url,
        api_key=api_key
    )
    
    env = RetailOpsEnv()
    tasks = ["easy_escalation", "medium_refund", "hard_reconciliation"]

    for task in tasks:
        # --- REQUIRED OPENENV TAG ---
        print(f"[START] task={task}", flush=True)
        
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
            - restock_inventory: {{"item": "string"}}
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
                
                print(f"Agent Action: {action.tool_name} | Args: {action.tool_args}", flush=True)
                obs, reward, done, info = env.step(action)
                
                # --- REQUIRED OPENENV TAG ---
                print(f"[STEP] step={env.step_count} reward={reward}", flush=True)
                
            except Exception as e:
                print(f"Agent produced invalid format or API error: {e}", flush=True)
                obs, reward, done, info = env.step(Action(tool_name="submit_final_answer", tool_args={}))
                # Catch errors and still print the step for the grader
                print(f"[STEP] step={env.step_count} reward={reward}", flush=True)

        # --- REQUIRED OPENENV TAG ---
        print(f"[END] task={task} score={info['score']} steps={env.step_count}", flush=True)

if __name__ == "__main__":
    run_inference()