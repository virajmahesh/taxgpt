from enum import Enum


class Providers(str, Enum):
    """
    API clients that can be used to generate embeddings.
    """

    OPENAI = "openai"
    TOGETHER = "together"
    COHERE = "cohere"
    HUGGING_FACE = "huggingface"