"""
    Functions to generate chat responses.

    Author: Viraj Mahesh (virajmahesh@gmail.com)
"""
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY
from providers import Providers


class ChatModel:
    """
    A model that can be used to generate chat responses.
    """

    name: str = None
    client: Any = None
    provider: str = None
    max_tokens: int = None

    @classmethod
    def chat_response(
        cls, messages: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, str]]]:
        """
        Generate a chat response using the model :cls:.

        :param messages: A list of messages. This should be in the OpenAI format
        [
            {"role": "<role>", "content": "<content>"},
            {"role": "<role>", "content": "<content>"}
            ...
        ]
        """
        response = cls.client.chat.completions.create(model=cls.name, messages=messages)
        response = response.choices[0].message
        return response.content, messages + [{response.role: response.content}]


openai_client = OpenAI(api_key=OPENAI_API_KEY)


class OpenAIModel(ChatModel):
    client = openai_client
    provider = Providers.OPENAI


class GPT432K(OpenAIModel):
    name = "gpt-4"
    max_tokens = 8000


class GPT4Turbo(OpenAIModel):
    name = "gpt-4-1106-preview"
    max_tokens = 128000


if __name__ == "__main__":
    print(GPT432K.chat_response([{"role": "user", "content": "Hello, how are you?"}]))
    print(GPT4Turbo.chat_response([{"role": "user", "content": "Hello, how are you?"}]))
