import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")  # chỉ rõ path, không phụ thuộc cwd

print("API KEY:", bool(os.getenv("FPT_API_KEY")))
print("BASE URL:", os.getenv("FPT_BASE_URL"))
print("MODEL:", os.getenv("FPT_LLM_MODEL"))

from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("FPT_API_KEY"),
    base_url=os.getenv("FPT_BASE_URL"),
)
resp = client.chat.completions.create(
    model=os.getenv("FPT_LLM_MODEL"),
    messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
    temperature=0.0,
)
print("Response:", resp.choices[0].message.content)
