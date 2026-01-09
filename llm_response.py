from openai import OpenAI

client = OpenAI()

def get_llm_response(user_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful voice assistant. "
                    "Always respond in English, even if the user speaks another language."
                )
            },
            {"role": "user", "content": user_text}
        ]
    )

    return response.choices[0].message.content
