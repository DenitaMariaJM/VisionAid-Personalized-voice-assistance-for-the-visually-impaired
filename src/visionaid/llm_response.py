from openai import OpenAI

client = OpenAI()

def get_llm_response(user_text, extra_context=""):
    if not user_text:
        return ""
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful voice assistant. "
                    "Always respond in English, even if the user speaks another language."
                ),
            }
        ]
        if extra_context:
            messages.append(
                {"role": "system", "content": f"Context:\n{extra_context}"}
            )
        messages.append({"role": "user", "content": user_text})
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        message = response.choices[0].message
        return (message.content or "").strip()
    except Exception as exc:
        print(f"LLM request failed: {exc}")
        return ""
