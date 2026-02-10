import asyncio
import os
import litellm
from src.llm_integration import LLMClient

async def check_error():
    print("Checking LLM Connectivity and Error details...")
    
    key_path = "api_key/openai_api.txt"
    if os.path.exists(key_path) and not os.environ.get("OPENAI_API_KEY"):
        with open(key_path, "r") as f:
            os.environ["OPENAI_API_KEY"] = f.read().strip()
            print(f"Loaded key from {key_path}")

    # Don't use the client yet, use litellm directly to avoid its internal retry logic
    messages = [{"role": "user", "content": "hi"}]
    
    print("\n--- Testing with model: gpt-5.2 ---")
    try:
        response = await litellm.acompletion(
            model="gpt-5.2",
            messages=messages,
            num_retries=0
        )
        print("Success!")
    except Exception as e:
        print(f"Caught Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        
    print("\n--- Testing with model: gpt-4o ---")
    try:
        response = await litellm.acompletion(
            model="gpt-4o",
            messages=messages,
            num_retries=0
        )
        print("Success!")
    except Exception as e:
        print(f"Caught Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")

if __name__ == "__main__":
    asyncio.run(check_error())