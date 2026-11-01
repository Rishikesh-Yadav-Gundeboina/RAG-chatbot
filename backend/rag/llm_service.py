from google import genai
from google.genai import types
from dotenv import load_dotenv
import httpx
import os

load_dotenv()
API_KEY = os.getenv('API_KEY')

LLM_TIMEOUT_SECONDS = 30


class LLMTimeoutError(Exception):
    """Raised when the LLM does not respond within LLM_TIMEOUT_SECONDS."""
    pass


class LLMService:
    def generate(self,prompt:str) -> str:
        client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(timeout=LLM_TIMEOUT_SECONDS * 1000),
        )
        try:
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt
            )
        except httpx.TimeoutException:
            raise LLMTimeoutError(f"LLM request timed out after {LLM_TIMEOUT_SECONDS} seconds")
        return response.text

if __name__ == "__main__":
    my_prompt = LLMService()
    my_result = my_prompt.generate("What is Numpy?")
    print(my_result)
