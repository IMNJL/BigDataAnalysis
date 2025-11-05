# main.py
import asyncio
from agent import run_agent
from inspector import print_examples

if __name__ == "__main__":
    input_type = input("Input type (topic/url): ").strip().lower()
    input_value = input("Enter topic or URL: ").strip()
    asyncio.run(run_agent(input_value, input_type))
    print_examples()
