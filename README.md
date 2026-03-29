# OpenEnv: RetailOpsEnv (E-Commerce Operations Agent)

## Environment Description & Motivation
`RetailOpsEnv` is a real-world simulation environment designed to train and evaluate AI agents on backend e-commerce operations. Instead of simulated games or toy grid-worlds, the agent acts as a Level-2 Customer Support Engineer. It must read incoming customer support tickets, query a mock internal database for order statuses, and execute state-changing operations to resolve the customer's issue. 

This environment tests an LLM's ability to ground its actions in retrieved data, correctly format strict JSON tool calls, sequence multiple dependent API actions, and recognize when to escalate out-of-bounds issues to a human manager.

---

## Action and Observation Spaces

### Observation Space (State)
The environment passes a strict JSON-serializable state to the agent at every step:
* **`ticket_text`** (str): The raw customer complaint or request.
* **`db_result`** (dict | null): The JSON payload returned from the most recent database query.
* **`system_message`** (str): Deterministic feedback from the environment detailing the success or failure of the last action (e.g., "Refund issued for 456", "Database queried successfully", "Invalid refund request").

### Action Space (Tools)
The agent must output a strict JSON object mapping to the OpenEnv-validated `Action` Pydantic model. 
Available tools and their required schemas:
* `query_database({"order_id": "string"})`: Fetches current order status and items.
* `issue_refund({"order_id": "string"})`: Processes a refund if the order exists and has not already been refunded.
* `restock_inventory({"item": "string"})`: Increments the inventory count for a returned or incorrectly shipped item.
* `escalate_ticket({})`: Immediately ends the episode and flags the ticket for human review.
* `submit_final_answer({})`: Ends the episode when the agent determines the task is fully resolved.

---

## Tasks & Baseline Scores
The environment includes three programmatic, deterministic graders that evaluate the final state of the mock database and the agent's action history. Scores are strictly bounded between `0.0` and `1.0`.

Baseline scores achieved using `llama-3.1-8b-instant` via the Groq API (max 10 steps per task):

1. **Easy Task (`easy_escalation`)**
   * **Scenario:** An angry customer demands to speak to a manager immediately.
   * **Expected Behavior:** Agent correctly identifies the tone and uses `escalate_ticket` without attempting to modify the database.
   * **Baseline Score:** `1.0 / 1.0`

2. **Medium Task (`medium_refund`)**
   * **Scenario:** Customer requests a refund for a damaged item (Order 456).
   * **Expected Behavior:** Agent queries the DB for order 456, verifies the order exists, and successfully issues a refund.
   * **Baseline Score:** `1.0 / 1.0`

3. **Hard Task (`hard_reconciliation`)**
   * **Scenario:** Customer received the wrong item (Order 789).
   * **Expected Behavior:** Agent must perform multi-step reconciliation: query the DB, issue a refund for the order, AND correctly extract the wrong item name to restock the inventory.
   * **Baseline Score:** `1.0 / 1.0`

---

## Setup & Usage Instructions

### 1. Local Python Execution
Install the dependencies:
```bash
pip install -r requirements.txt


Set the required OpenEnv variables in your terminal. (The below example uses Groq, but any OpenAI-compatible endpoint works):

Bash

# Windows PowerShell
$env:API_BASE_URL="[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)"
$env:MODEL_NAME="llama-3.1-8b-instant"
$env:HF_TOKEN="your_api_key_here"

# Mac/Linux
export API_BASE_URL="[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)"
export MODEL_NAME="llama-3.1-8b-instant"
export HF_TOKEN="your_api_key_here"
Run the baseline inference script to verify the agent and graders:

Bash

python inference.py
2. Docker / Hugging Face Spaces Deployment
This project includes a Dockerfile and server.py designed to meet the Hugging Face Spaces health-check requirements.

To build and run the containerized environment locally:

Bash

docker build -t retailops-env .
docker run -p 7860:7860 retailops-env
Visit http://localhost:7860/ in your browser. A 200 OK response confirms the environment is active and ready for automated validation.