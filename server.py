from fastapi import FastAPI, Request
from env import RetailOpsEnv
from models import Action

app = FastAPI()
env = RetailOpsEnv()

@app.get("/")
def health_check():
    return {"status": "ok", "environment": "RetailOpsEnv active"}

@app.post("/reset")
async def reset_env(request: Request):
    # The automated grader sends parameters (like task_id) in the POST body.
    # This safely accepts the JSON without throwing a strict 422 Error.
    try:
        body = await request.json()
        task_id = body.get("task_id", "easy_escalation")
    except:
        task_id = "easy_escalation"
        
    obs = env.reset(task_id=task_id)
    return obs.model_dump()

# The grader may also want to verify the step and state endpoints exist
@app.post("/step")
async def step_env(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state_env():
    return env.state()