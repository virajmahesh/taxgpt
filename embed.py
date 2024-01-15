"""
    Functions and classes to generate text embeddings.

    Author: Viraj Mahesh (virajmahesh@gmail.com)
"""
import tiktoken
import together
import numpy as np
from enum import Enum
from typing import Any

from config import *

openai_encoder = tiktoken.encoding_for_model(OPENAI_EMBED_MODEL)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

together.api_key = TOGETHER_API_KEY
together_client = together.Together()


class Providers(str, Enum):
    """
    API clients that can be used to generate embeddings.
    """

    OPENAI = "openai"
    TOGETHER = "together"


class EmbeddingModel:
    """
    A model that can be used to generate embeddings.
    """

    name: str = None
    client: Any = None
    provider: str = None
    max_tokens: int = None

    @classmethod
    def embed_text(cls, s: str) -> np.ndarray:
        """
        Embed :s: using the model :cls:.

        :param s: The text to embed.
        :return: The embedding of :s:.
        """
        response = cls.client.embeddings.create(input=s, model=cls.name)
        return np.asarray(response.data[0].embedding)


class OpenAIADA8K(EmbeddingModel):
    name = "text-embedding-ada-002"
    client = openai_client
    provider = Providers.OPENAI
    max_tokens = 8096


class Together32K(EmbeddingModel):
    name = "togethercomputer/m2-bert-80M-32k-retrieval"
    client = together_client
    provider = Providers.TOGETHER
    max_tokens = 32384


class Together8K(EmbeddingModel):
    name = "togethercomputer/m2-bert-80M-8k-retrieval"
    client = together_client
    provider = Providers.TOGETHER
    max_tokens = 8192


class UAELargeV1(EmbeddingModel):
    name = "WhereIsAI/UAE-Large-V1"
    client = together_client
    provider = Providers.TOGETHER
    max_tokens = 384  # Use a slightly smaller chunk size


def chunk_length(c: str) -> int:
    """
    Returns the length of :c: in tokens. By default, this uses
    the OpenAI encoder.

    #TODO: Move this to a method in EmbeddingModel, which uses different
    encoders depending on the model.
    """
    return len(openai_encoder.encode(c))


def split_text_to_chunks(text: str, chunk_size: int) -> list:
    """
    Split :text: into chunks of size :chunk_size:. Text is split line by line.

    #TODO: Add support for overlapping sentences.
    """
    result = []
    chunk = ""

    # Iterate through each line in the text
    for c in text.split("\n"):
        c = c.strip()
        if c == "":
            continue

        # If adding the next line makes the chunk too long, break the chunk
        # at this point and start a new one.
        if chunk_length(chunk + c) > chunk_size:
            result.append(chunk)
            chunk = c
        else:
            chunk += c

    # Add the last chunk
    result.append(chunk)
    return result


def similiarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
