import os
import logging
import datetime
import asyncio
import google.cloud.logging
from google.cloud import datastore
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from mcp.server.fastmcp import FastMCP 

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext

# --- 1. Setup Logging ---
try:
    cloud_logging_client = google.cloud.logging.Client()
    cloud_logging_client.setup_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)

load_dotenv()
model_name = os.getenv("MODEL", "gemini-1.5-flash")

# --- 2. Database Setup ---
# PRO TIP: For the default database, leaving arguments empty is the most stable 
# way to deploy on Google Cloud. It auto-detects the project and (default) DB.
# We use the DB_ID variable defined as a string to avoid 'name not defined' errors.
DB_ID = "sachin" 

try:
    db = datastore.Client(project=DB_ID) 
except Exception as e:
    logging.warning(f"Could not connect to Datastore: {e}")
    # Fallback to default client if project ID is auto-detected by gcloud
    db = datastore.Client()

mcp = FastMCP("WorkspaceTools")

# ================= 3. TOOLS =================

@mcp.tool()
def add_task(title: str) -> str:
    """Adds a new task to the workspace."""
    try:
        key = db.key('Task')
        task = datastore.Entity(key=key)
        task.update({
            'title': title, 
            'completed': False, 
            'created_at': datetime.datetime.now()
        })
        db.put(task)
        return f"Success: Task '{title}' saved (ID: {task.key.id})."
    except Exception as e:
        logging.error(f"DB Error in add_task: {e}")
        return f"Database Error: {str(e)}"

@mcp.tool()
def list_tasks() -> str:
    """Lists all current tasks."""
    try:
        query = db.query(kind='Task')
        tasks = list(query.fetch())
        if not tasks: return "Your task list is empty."
        
        res = ["📋 Current Tasks:"]
        for t in tasks:
            status = "✅" if t.get('completed') else "⏳"
            res.append(f"{status} {t.get('title')} (ID: {t.key.id})")
        return "\n".join(res)
    except Exception as e:
        return f"Database Error: {str(e)}"

@mcp.tool()
def complete_task(task_id: str) -> str:
    """Marks a task as complete. Input must be the numeric ID."""
    try:
        numeric_id = int(''.join(filter(str.isdigit, task_id)))
        key = db.key('Task', numeric_id)
        task = db.get(key)
        if task:
            task['completed'] = True
            db.put(task)
            return f"Task {numeric_id} marked as done."
        return f"Task {numeric_id} not found."
    except Exception as e:
        return f"Error processing task ID: {str(e)}"

@mcp.tool()
def add_note(title: str, content: str) -> str:
    
    try:
        key = db.key('Note')
        note = datastore.Entity(key=key)
        note.update({'title': title, 'content': content, 'at': datetime.datetime.now()})
        db.put(note)
        return f"Note '{title}' saved successfully."
    except Exception as e:
        return f"Database Error: {str(e)}"

# ================= 4. AGENTS =================

def add_prompt_to_state(tool_context: ToolContext, prompt: str):
    """Internal tool to bridge user intent across the agent workflow."""
    tool_context.state["PROMPT"] = prompt
    return {"status": "ok"}

def workspace_instruction(ctx):
    # Pull the prompt from the state
    user_prompt = ctx.state.get("PROMPT", "Welcome the user.")
    return f"""
You are the Workspace Executive Assistant for Sachin. 
CRITICAL IDENTITY RULE: The user's name is Sachin. Always address the user as Sachin. 
Never use the name 'Dr. Abhishek' or any other name.

Greeting: Always start with a polite, professional greeting addressing Sachin.
Task: Use your tools to complete this specific request: {user_prompt}
"""

def root_instruction(ctx):
    # Pulls the prompt directly from the API call
    raw_input = ctx.state.get("user_input", "Hello")
    return f"""
The user's name is Sachin. 
1. Save this user input using 'add_prompt_to_state': {raw_input}
2. Hand off control to the 'workflow' agent immediately.
"""

workspace_agent = Agent(
    name="workspace",
    model=model_name,
    instruction=workspace_instruction,
    tools=[add_task, list_tasks, complete_task, add_note]
)

workflow = SequentialAgent(
    name="workflow",
    sub_agents=[workspace_agent]
)

root_agent = Agent(
    name="root",
    model=model_name,
    instruction=root_instruction,
    tools=[add_prompt_to_state],
    sub_agents=[workflow]
)

# ================= 5. API =================

app = FastAPI()

class UserRequest(BaseModel):
    prompt: str

@app.post("/api/v1/workspace/chat")
async def chat(request: UserRequest):
    try:
        final_reply = ""
        # Inject user_input into the agent state
        async for event in root_agent.run_async({"user_input": request.prompt}):
            if hasattr(event, 'text') and event.text:
                final_reply = event.text

        return {
            "status": "success",
            "reply": final_reply if final_reply else "Request processed."
        }

    except Exception as e:
        logging.error(f"Chat Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)