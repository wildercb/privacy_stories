import openai
import json

class ResponseHandler:
    def __init__(self, model_name):
        self.client = openai.OpenAI()
        self.model_name = model_name

    def get_response(self, prompt, build_string):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": build_string},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
