from src.module3_llm.llm_client import RetailLLMClient
from src.module3_llm.query_engine import QueryEngine

client = RetailLLMClient()
engine = QueryEngine(llm_client=client)

questions = [
    "Which customers are about to leave?",
    "What products need reordering?",
    "Show me revenue at risk and low stock items",
    "What is the weather today?"
]

for q in questions:
    route = engine._route_question(q)
    print(f"Q: {q}")
    print(f"-> Route: {route}\n")
    

from src.module3_llm.llm_client import RetailLLMClient
from src.module3_llm.query_engine import QueryEngine

client = RetailLLMClient()
engine = QueryEngine(llm_client=client)

test_questions = [
    "Which customers are at highest risk of leaving?",
    "What products need reordering urgently?",
    "Give me an overall business health summary",
    "What is the weather forecast for London?"
]

for q in test_questions:
    result = engine.answer(q)
    print(f"Q: {result['question']}")
    print(f"Route: {result['route']}")
    print(f"A: {result['answer']}")
    print()