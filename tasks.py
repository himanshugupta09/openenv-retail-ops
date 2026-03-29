TASKS = {
    "easy_escalation": {
        "ticket": "I am furious! Your support is terrible. Connect me to a manager right now!",
        "initial_db": {"123": {"status": "shipped", "item": "laptop"}}
    },
    "medium_refund": {
        "ticket": "Hi, order 456 arrived completely smashed. I want a refund.",
        "initial_db": {"456": {"status": "delivered", "item": "vase", "refunded": False}}
    },
    "hard_reconciliation": {
        "ticket": "I got order 789, but it's the wrong item. Please refund me and fix your system.",
        "initial_db": {
            "789": {"status": "delivered", "item": "wrong_item", "refunded": False},
            "inventory": {"wrong_item": 5}
        }
    }
}

def grade_task(task_id: str, final_state: dict, action_history: list) -> float:
    if task_id == "easy_escalation":
        if action_history and action_history[-1]["tool_name"] == "escalate_ticket":
            return 1.0
        return 0.0

    elif task_id == "medium_refund":
        db = final_state.get("db", {})
        if db.get("456", {}).get("refunded") is True:
            return 1.0
        return 0.0

    elif task_id == "hard_reconciliation":
        score = 0.0
        db = final_state.get("db", {})
        if db.get("789", {}).get("refunded") is True:
            score += 0.5
        if db.get("inventory", {}).get("wrong_item") == 6: 
            score += 0.5
        return score
        
    return 0.0