from google.genai import types
import json

# Класс для обращения к Open AI API
class GPTAIAgent:
    def __init__(self, client):

        self.client = client

    async def run(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:

        self.response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(user_prompt,schema)},
            ],
            temperature=0.2
        )
        self.message = self.response.choices[0].message.content
        return json.loads(self.message)

class GeminiAIAgent:
    def __init__(self, client):
        """
        Args:
            client: Google Cloud Platform Client
        """
        self.client = client

    async def run(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        """
        Args:
            system_prompt: system level prompt
            user_prompt: user prompt
            schema: data
        Returns:
            dict: agent response
        """
        try:
            self.response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=build_user_prompt(user_prompt,schema),
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
            )
            )
        except GoogleAPIError as e:
            # Ошибки API (ключ, квоты, доступ)
            self.response = {
                "error": "API_ERROR",
                "message": str(e)
            }
        except Exception as e:
            # Всё остальное (баги, сеть, код)
            self.response = {
                "error": "UNKNOWN_ERROR",
                "message": str(e)
            }


        self.message = self.response.text
        return json.loads(self.message)


def build_user_prompt(user_prompt:str,schema:dict) -> str:
    return f"""Входные данные
{schema}
Задача:
{user_prompt}"""


SYSTEM_PROMPT = """
Ты backend-сервис.
Ты отвечаешь ТОЛЬКО валидным JSON.
Никакого текста вне JSON.
Все поля должны строго соответствовать схеме.
Если данных недостаточно — используй разумные предположения.
"""

DESCRIPTION_PROMPT = """
Сформируй JSON краткое описание задачи в строку string. Допустимый вид:
{
  "description": "string"
}
"""

DEADLINE_PROMPT = """ 
Сформируй в JSON дату дедлайна в виде YYYY-MM-DD и короткие причины того что дедлайн такой строкой string. Допустимый вид:
{
  "deadline": "YYYY-MM-DD"",
  "reasoning": "string"
}
"""

PRIORITY_PROMPT = """
Сформируй JSON  выбери один из приоритетов ["low", "medium", "high"] и обоснуй выбор в строке string. Допустимый вид:
{
  "priority": ["low", "medium", "high"],
  "reasoning": "string"
} """

CATEGORY_PROMPT = """ 
Сформируй JSON и опиши на основе входных данных категорию задачи и список одно-двух-словных тэгов этой задачи. Допустимый вид:
{
    "category": "improvement",
    "tags":  ["tag1","tag2","tag3"]
}
"""

COMMON_PROMPT = """
Сформируй JSON на основе предыдущих данных. Что ты должен написать в каждом поле:
description - короткое. однострочное описание задачи.
priority - выбери приоритет задачи.
deadline - дата до которой должно быть выполненно задание
category - категория задачи.
tags - сформиируй список тегов для задачи.
Допустимый вид:
 {
  "description": "string",
  "priority": ["low", "medium", "high"],
  "deadline": "YYYY-MM-DD",
  "category": "string",
  "tags": ["tag1","tag2","tag3"]"
}
 """