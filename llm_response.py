from openai import OpenAI

client = OpenAI()

def get_llm_response(user_text):
    if not user_text:
        return ""
    try:
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
        message = response.choices[0].message
        return (message.content or "").strip()
    except Exception as exc:
        print(f"LLM request failed: {exc}")
        return ""
