"""
    Functions and classes to generate text embeddings.

    Author: Viraj Mahesh (virajmahesh@gmail.com)
"""
from enum import Enum
from typing import Any

import cohere
import numpy as np
import tiktoken
import together
from openai import OpenAI

from config import (COHERE_API_KEY, OPENAI_API_KEY, OPENAI_EMBED_MODEL,
                    TOGETHER_API_KEY)
from providers import Providers

openai_encoder = tiktoken.encoding_for_model(OPENAI_EMBED_MODEL)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

together.api_key = TOGETHER_API_KEY
together_client = together.Together()

cohere_client = cohere.Client(COHERE_API_KEY)


class InputTypes(str, Enum):
    """
    Types of inputs that can be used to generate embeddings.
    """

    SEARCH_DOCUMENT = "search_document"
    SEARCH_QUERY = "search_query"


class EmbeddingModel:
    """
    A model that can be used to generate embeddings.
    """

    name: str = None
    client: Any = None
    provider: str = None
    max_tokens: int = None

    @classmethod
    def embed_text(cls, s: str, **kwargs) -> np.ndarray:
        """
        Embed :s: using the model :cls:.

        :param s: The text to embed.
        :param kwargs: Arbitrary keyword arguments.
        :return: The embedding of :s:.
        """
        response = cls.client.embeddings.create(input=s, model=cls.name)
        return np.asarray(response.data[0].embedding)

    @classmethod
    def file_safe_name(cls) -> str:
        return cls.name.replace("/", "-")


class OpenAIADA8K(EmbeddingModel):
    name = "text-embedding-ada-002"
    client = openai_client
    provider = Providers.OPENAI
    max_tokens = 8000


class TogetherEmbeddingModel(EmbeddingModel):
    client = together_client
    provider = Providers.TOGETHER


class Together32K(TogetherEmbeddingModel):
    name = "togethercomputer/m2-bert-80M-32k-retrieval"
    max_tokens = 32000


class Together8K(TogetherEmbeddingModel):
    name = "togethercomputer/m2-bert-80M-8k-retrieval"
    max_tokens = 8000


class UAELarge(TogetherEmbeddingModel):
    name = "WhereIsAI/UAE-Large-V1"
    max_tokens = 384  # Use a slightly smaller chunk size


class BAAILarge(TogetherEmbeddingModel):
    name = "BAAI/bge-large-en-v1.5"
    max_tokens = 384  # Use a slightly smaller chunk size


class MistralLarge(EmbeddingModel):
    """
    Embedding model from Mistral. Requires a custom embed_text method.
    """

    pass


class Cohere(EmbeddingModel):
    """
    Embedding model from Cohere. Requires a custom embed_text method.
    """

    name = "embed-english-v3.0"
    provider = Providers.COHERE
    client = cohere_client
    max_tokens = 384

    @classmethod
    def embed_text(cls, s: str, **kwargs) -> np.ndarray:
        response = cls.client.embed(
            texts=[s],
            model=cls.name,
            input_type=kwargs.get("input_type"),
        )
        return np.asarray(response.embeddings[0])


def chunk_length(c: str) -> int:
    """
    Returns the length of :c: in tokens. By default, this uses
    the OpenAI encoder.

    #TODO: Move this to a method in EmbeddingModel, which uses different
    encoders depending on the model.
    """
    return len(openai_encoder.encode(c))


def encode_text(text: str) -> list:
    return openai_encoder.encode(text)


def decode_single_token_str(token: int) -> str:
    return openai_encoder.decode_single_token_bytes(token).decode("utf-8")


def tokenize_text(text: str) -> list:
    """
    Tokenize :text: using the OpenAI encoder.
    """
    tokens = encode_text(text)
    tokens = map(decode_single_token_str, tokens)
    return list(tokens)


def split_text_to_chunks(text: str, chunk_size: int) -> list:
    """
    Split :text: into chunks of size :chunk_size:. Text is split line by line.

    #TODO: Add support for overlapping sentences.
    """
    result = []
    chunk = ""

    # Iterate through each line in the text
    tokens = tokenize_text(text)
    for c in tokens:
        if c.strip() == "":
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
