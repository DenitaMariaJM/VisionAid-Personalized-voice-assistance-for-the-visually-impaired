from openai import OpenAI

# Will read from environment variable OPENAI_API_KEY
client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input="Hello! Can you confirm the API is working?"
)

print("Model reply:", response.output_text)
