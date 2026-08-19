from BackEnd.app import services
from BackEnd.app import services
from pathlib import Path
import json
from dotenv import load_dotenv
import os 
import asyncio
import httpx
from BackEnd.CONFIG import RANKING_MODEL

load_dotenv()

LLM_API = os.getenv("RANKING_MODEL_API")
url = "https://mkp-api.fptcloud.com/chat/completions"
path = Path.cwd() / "BackEnd" / "app" / "Fusion" / "prompt.txt"

class LLM_DynamicWeight: 
    def __init__(self):
        self.model = RANKING_MODEL
        with open(path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        self.header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API}"
        }

    async def dynamic_weight(self, user_prompt: str): 
        data = {
            "model": f"{self.model}",                           # Model name

            "messages": [                                      # List of message objects. Please update the System prompt to have the model respond appropriately
                {
                    "role": "system",
                    "content": f"{self.system_prompt}"
                },
                {
                    "role": "user",
                    "content": f"{user_prompt}"
                }
            ]   
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=url,
                json=data,
                headers=self.header
            )

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return json.loads(content)

async def main(): 
    llm = LLM_DynamicWeight()
    result = await llm.dynamic_weight("Tìm cho tôi hình ảnh có 1 gia đình ngồi tụ họp")
    print(result)

    
if __name__ == "__main__":
    asyncio.run(main())