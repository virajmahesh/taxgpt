"""
    This file contains utility functions to process the tax code and generate embeddings.
    We assume the tax code has already been downloaded in XHTML format 
    from https://uscode.house.gov/download/download.shtml.

    Author: Viraj Mahesh (virajmahesh@gmail.com)
"""

import os
import re
import json
from sqlalchemy import Engine
from config import *
import numpy as np
from tqdm import tqdm
from enum import Enum
from typing import Any
from openai import OpenAI
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlmodel import SQLModel, Field, create_engine, Session, select


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


    def embed_text(cls, s: str):
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


class Statute(SQLModel, table=True):
    """
    A model for a statute in the US tax code.
    """

    id: int = Field(default=None, primary_key=True)
    section: str = Field(default=None)
    title: str = Field(default=None)
    text: str = Field(default=None)
    token_count: int = Field(default=0)


class Embedding(SQLModel, table=True):
    """
    A model representing a chunk of text from a statute and it's embedding.

    Primary key: (id, model, provider)
    """

    id: int = Field(default=None, primary_key=True)
    statute_id: int = Field(default=None, foreign_key="statute.id")
    text: str = Field(default=None)
    token_count: int = Field(default=0)
    model: str = Field(default=None)
    provider: str = Field(default=None)
    embedding_vector: str = Field(default=None)


def split_tax_code(path: str = TAX_CODE_XHTML_PATH, out_dir=DATA_DIR) -> None:
    """
    Split XHTML tax code at :path: into multiple files using BeautifulSoup.
    Creates one for each section of the tax code named {section_number}.txt

    :param path: The path to the tax code XHTML file. By default, this is
                 TAX_CODE_XHTML_PATH.
    """

    def find_fields(field_name: str, html: str) -> list:
        """
        Find all fields in :html: with name :field_name:.

        :param field_name: The name of the field to find.
        :param html: The HTML to search in.
        :return: A list of all fields with name :field_name:.
        """
        pattern = (
            f"<!-- field-start:{field_name} -->(.*?)<!-- field-end:{field_name} -->"
        )
        return re.findall(pattern, html, re.DOTALL)

    def get_text_from_html(html: str) -> str:
        """
        Convert :html: to plain text using BeautifulSoup.
        """
        return BeautifulSoup(html, "html.parser").text

    with open(path, "r") as f:
        html = f.read()

        headings = find_fields("head", html)
        statute_bodies = find_fields("statute", html)

        for i, (h, s) in enumerate(zip(headings, statute_bodies)):
            with open(f"{out_dir}/{i}.txt", "w") as f:
                text = get_text_from_html(h)
                text = text.lstrip().rstrip()
                f.write(text)
                f.write("\n")
                f.write(get_text_from_html(s))
                f.close()


def load_into_db(path: str = DATA_DIR) -> None:
    """
    Load the statute text in directory specified by :path: into a SQLite database.
    """

    def get_section_and_title(first_line: str) -> tuple:
        """
        Extract the section number and title from the first line of the statute.
        """
        result = re.search(r"§(\S+\.\S+) (.*)", first_line)
        if result is None:
            result = re.search(r"§(\S+)\. (.*)", first_line)
        section, title = result.groups()
        title = title.strip()
        return section, title

    # Create the database and table
    engine = create_engine(SQL_ENGINE_PATH)
    SQLModel.metadata.create_all(engine)

    # Load the statutes and add them to the table
    with Session(engine) as session:
        # Get a list of .txt files in :path:
        files = [f for f in os.listdir(path) if f.endswith(".txt")]

        # Convert the file name to an integer for sorting. Select
        # everything before the period and convert it to an integer.
        file_name_to_text = lambda x: int(x.split(".")[0])

        # Load each of the files into the database
        for f in sorted(files, key=file_name_to_text):
            with open(f"{path}/{f}", "r") as statute_file:
                first_line = statute_file.readline()

                # Extract the section number and title from the first line.
                # The first line of the file looks like §{section_number} {section_title}
                section, title = get_section_and_title(first_line)

                # Read the rest of the file, and remove any leading or trailing whitespace
                text = statute_file.read().strip()
                token_count = len(openai_encoder.encode(text))
                statute = Statute(
                    section=section, title=title, text=text, token_count=token_count
                )

                # Add the statute to the database
                session.add(statute)
        session.commit()


def visualize_token_count() -> None:
    """
    Visualize the distribution of token counts in the statutes.
    """
    engine = create_engine(SQL_ENGINE_PATH)
    with Session(engine) as session:
        statutes = session.exec(select(Statute)).all()
        token_counts = [s.token_count for s in statutes]
        plt.hist(token_counts, bins=100)
        plt.xlabel("Token Count")
        plt.ylabel("Frequency")
        plt.title("Distribution of Token Counts in Statutes")
        plt.show()


def chunk_length(c: str) -> int:
    """Returns the length of a chunk in tokens."""
    return len(openai_encoder.encode(c))


def split_text_to_chunks(text: str, chunk_size: int) -> list:
    """
    Split :text: into chunks of size :chunk_size:. Text is split line by line.
    """
    result = []
    chunk = ""
    for c in text.split("\n"):
        c = c.strip()
        if c == "":
            continue

        # If adding the next line makes the chunk too long, break the chunk at this
        # point and start a new one.
        if chunk_length(chunk + c) > chunk_size:
            result.append(chunk)
            chunk = c
        else:
            chunk += c

    # Add the last chunk
    result.append(chunk)
    return result


def embed_text(s: str, client: Any, model: str) -> np.ndarray:
    """
    Generate a vector embedding for :s: using the specified client and model.
    """
    response = client.embeddings.create(input=s, model=model)
    return np.asarray(response.data[0].embedding)


def similiarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embed_statutes(model: EmbeddingModel, offset: int = 0) -> None:
    """
    Embed all statutes using the OpenAI API. The statutes are split into chunks
    of 8096 tokens, and each chunk is embedded separately.

    :model: The model to use to generate the embeddings.

    :offset: The offset to start at. This is useful if the embedding process
    is interrupted, and needs to be restarted from a specific point.
    """

    def embed_statute(s, model: EmbeddingModel, engine: Engine):
        """
        Embed :s: using the specified model.

        :s: The statute to embed.
        :model: The model to use to generate the embeddings.
        :progress: A tqdm object to track progress.

        :return: The id of the next chunk.
        """
        # Skip empty statutes
        if s.text.strip() == "":
            return

        chunks = split_text_to_chunks(s.text, model.max_tokens)

        with Session(engine) as session:
            # Iterate through each chunk and embed it
            for c in chunks:
                e = embed_text(c, model.client, model.name)
                embedding = Embedding(
                    statute_id=s.id,
                    text=c,
                    token_count=chunk_length(c),
                    model=model.name,
                    provider=model.provider,
                    embedding_vector=json.dumps(e.tolist()),
                )
                session.add(embedding)
            session.commit()  # Only commit once the entire statute has been embdded

    engine = create_engine(SQL_ENGINE_PATH)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        statutes = session.exec(select(Statute).offset(offset)).all()

        # Generate embeddings for all statutes
        # Each thread is given one full statute to deal with.
        with ThreadPoolExecutor(max_workers=10) as t:
            with tqdm(total=len(statutes)) as progress:
                futures = [t.submit(embed_statute, s, model, engine) for s in statutes]
                for _ in as_completed(futures):
                    progress.update(1)


def generate_embedding_dump(
    path: str = DATA_DIR, model: EmbeddingModel = OpenAIADA8K
) -> None:
    """
    Loads the generated embeddings from the database, packs them into a numpy
    matrix and saves them to a file.

    :path: The directory to save the embeddings to.
    """
    engine = create_engine(SQL_ENGINE_PATH)
    with Session(engine) as session:
        embeddings = session.exec(
            select(Embedding).where(Embedding.provider == model.provider).order_by(Embedding.id)
        ).all()

        # Generate a list of embeddings
        embedding_list = []
        for e in embeddings:
            embedding_list.append(json.loads(e.embedding_vector))

        query = """What is the personal income tax rate?"""

        e1 = embed_text(query, model.client, model.name)
        e1 /= np.linalg.norm(e1)
        print(e1.shape)

        # Convert the list to a numpy array
        embedding_matrix = np.asarray(embedding_list).T
        embedding_matrix /= np.linalg.norm(embedding_matrix, axis=0)
        print(embedding_matrix.shape)

        # Print the indices of the most similar statutes to e1, in descending
        # order of similarity
        similarities = np.dot(e1, embedding_matrix)
        indices = np.argsort(similarities)[::-1]
        chunk_id = np.asarray([e.id for e in embeddings])
        print(indices[:100])
        print(chunk_id[indices[:100]])
        print(similarities[indices[:100]])
        print(embeddings[indices[0]].text)

        # Save the matrix to a file
        np.save(f"{path}/together_embeddings.npy", embedding_matrix)


if __name__ == "__main__":
    #load_into_db()
    #embed_statutes(model=Together8K)
    generate_embedding_dump(model=Together8K)
