from fastapi import FastAPI
from env import RetailOpsEnv

app = FastAPI()
env = RetailOpsEnv()

@app.get("/")
def health_check():
    return {"status": "ok", "environment": "RetailOpsEnv active"}

@app.post("/reset")
def reset_env():
    obs = env.reset()
    return obs.model_dump()