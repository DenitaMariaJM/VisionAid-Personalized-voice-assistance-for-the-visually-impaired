from openai import OpenAI
import base64

client = OpenAI()
IMAGE_PATH = "captured_images/test.jpg"

img = base64.b64encode(open(IMAGE_PATH, "rb").read()).decode()

resp = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {"type": "input_text", "text": "Describe this image"},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img}"}
    ]
)

print(resp)
