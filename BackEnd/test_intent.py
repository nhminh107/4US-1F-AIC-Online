import asyncio
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from BackEnd.app.contracts.models import RawQuery
from BackEnd.app.intent_extractor.extractor import extract_intent

async def main():
    print("=== TEST KIS ===")
    q1 = RawQuery(text="Tìm cảnh người đàn ông áo đỏ đứng cạnh bảng HCMC")
    r1 = await extract_intent(q1)
    print(r1.model_dump_json(indent=2))

    print("\n=== TEST VQA ===")
    q2 = RawQuery(text="Người phụ nữ trong video đang làm gì?")
    r2 = await extract_intent(q2)
    print(r2.model_dump_json(indent=2))

asyncio.run(main())
