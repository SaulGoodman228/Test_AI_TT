from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import TitleModel, DescriptionModel

from ai_agents import (GeminiAIAgent,
                       SYSTEM_PROMPT, DESCRIPTION_PROMPT,DEADLINE_PROMPT,
                       PRIORITY_PROMPT, CATEGORY_PROMPT,COMMON_PROMPT)

from openai import OpenAI
from google import genai

from typing import Dict
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app = FastAPI()
agent = GeminiAIAgent(client)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы (GET, POST, OPTIONS и т.д.)
    allow_headers=["*"],  # Разрешить все заголовки
)

"""1. Descriptiion"""
@app.post('/ai/generate-description')
async def generate_description(title: TitleModel) -> Dict:
    """
    Args:
        title: task name
    Returns:
        message: task description
    """
    message = await agent.run(SYSTEM_PROMPT,DESCRIPTION_PROMPT,title.model_dump())
    return message

"""2. deadline/reason"""
@app.post('/ai/suggest-deadline')
async def suggest_deadline(description: DescriptionModel) -> Dict:
    """
    Args:
        description: task name and description
    Returns:
        message: task deadline and reasons
    """
    message = await agent.run(SYSTEM_PROMPT, DEADLINE_PROMPT, description.model_dump())
    return message

"""3. priority/reason"""
@app.post("/ai/analyze-priority")
async def analyze_priority(description: DescriptionModel) -> Dict:
    """
    Args:
        description: task name and description
    Returns:
        message: task priority and reasons
    """
    message = await agent.run(SYSTEM_PROMPT, PRIORITY_PROMPT, description.model_dump())
    return message

"""4. category"""
@app.post("/ai/categorize")
async def categorize(description: DescriptionModel) -> Dict:
    """
    Args:
        description: task name and description
    Returns:
        message: task category and tags
    """
    message = await agent.run(SYSTEM_PROMPT, CATEGORY_PROMPT, description.model_dump())
    return message

"""5. description/priority/deadline/category/tags"""
@app.post("/ai/process-task")
async def process_task(title: TitleModel) -> Dict:
    """
    Args:
        title: task name and description
    Returns:
        message: all task data
    """
    message = await agent.run(SYSTEM_PROMPT,COMMON_PROMPT,title.model_dump())
    return message
