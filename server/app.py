import uvicorn
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

def main():
    # Updated to point to the new folder structure
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()