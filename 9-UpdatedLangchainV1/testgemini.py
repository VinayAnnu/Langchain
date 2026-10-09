import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


# .env is one directory above this project
load_dotenv("../.env")

print("API key loaded:", bool(os.getenv("GOOGLE_API_KEY")))

client = genai.Client()

print("Client created")

response = client.models.generate_content(
    model="gemini-3.7-flash",
    contents="Say hello in one sentence.",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="low"
        )
    )
)

print("Response received")
print(response.text)