import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

model = 'gemma-4-31b-it'
prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

response = client.models.generate_content(
    model=model, contents=prompt
)

prompt_tokens = response.usage_metadata.prompt_token_count
response_tokens = response.usage_metadata.candidates_token_count

print(response.text)
print(f'Prompt tokens: {prompt_tokens}')
print(f'Response tokens: {response_tokens}')