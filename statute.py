"""
    This file contains utility functions to process the tax code and generate embeddings.
    We assume the tax code has already been downloaded in XHTML format 
    from https://uscode.house.gov/download/download.shtml.

    Author: Viraj Mahesh (virajmahesh@gmail.com)
"""

import os
import re
import json
import tiktoken
import together
import numpy as np
from tqdm import tqdm
from enum import Enum
from openai import OpenAI
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from sqlmodel import SQLModel, Field, create_engine, Session, select, ARRAY
from typing import Any


#########################
# Constants and Globals #
#########################

# File paths
DATA_DIR = "data"
TAX_CODE_XHTML_PATH = f"{DATA_DIR}/title26.htm"

# OpenAI Config
OPENAI_EMBED_MODEL = "text-embedding-ada-002"  # Set this explicitly so that it doesn't change automatically
openai_encoder = tiktoken.encoding_for_model(OPENAI_EMBED_MODEL)
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Together Config
TOGETHER_EMBED_MODEL = "togethercomputer/m2-bert-80M-32k-retrieval"
together.api_key = os.environ["TOGETHER_API_KEY"]
together_client = together.Together()


# SQL Enginge for SQLModel
SQL_ENGINE = "sqlite"
DB_PATH = "statute.db"
SQL_ENGINE_PATH = f"{SQL_ENGINE}:///{DB_PATH}"


class Clients(str, Enum):
    """
    A list of API clients that can be used to generate embeddings.
    """

    OPENAI = "openai"
    TOGETHER = "together"


class Models(str, Enum):
    """
    A list of models that can be used to generate embeddings.
    """

    OPENAI_ADA_8K = "text-embedding-ada-002"
    TOGETHER_M2_32K = "togethercomputer/m2-bert-80M-32k-retrieval"


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
    """

    id: int = Field(
        default=None, primary_key=True
    )  # Uniquely identifies a chunk-statute pair
    statute_id: int = Field(default=None, foreign_key="statute.id")
    text: str = Field(default=None)
    token_count: int = Field(default=0)
    model: str = Field(default=None, primary_key=True)
    client: str = Field(default=None, primary_key=True)
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


def embed_statutes_together_API(offset: int = 0):
    """
    Embed all statutes using the Together API.

    :offset: The offset to start at. This is useful if the embedding process
    is interrupted, and needs to be restarted from a specific point.
    """
    engine = create_engine(SQL_ENGINE_PATH)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        statutes = session.exec(select(Statute).offset(offset)).all()

        # Generate embeddings for all statutes
        for i, s in enumerate(tqdm(statutes)):  # TODO: Print progress to the terminal
            # Skip empty statutes
            if s.text.strip() == "":
                continue

            e = embed_text(s.text, together_client, TOGETHER_EMBED_MODEL)
            embedding = Embedding(
                id=i,
                statute_id=s.id,
                text=s.text,
                token_count=s.token_count,
                model=Models.TOGETHER_M2_32K,
                client=Clients.TOGETHER,
                embedding_vector=json.dumps(e.tolist()),
            )
            session.add(embedding)
            session.commit()


def embed_statutes_openai_API(offset: int = 0):
    """
    Embed all statutes using the OpenAI API. The statutes are split into chunks
    of 8096 tokens, and each chunk is embedded separately.

    :offset: The offset to start at. This is useful if the embedding process
    is interrupted, and needs to be restarted from a specific point.
    """
    engine = create_engine(SQL_ENGINE_PATH)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        statutes = session.exec(select(Statute).offset(offset)).all()

        # Generate embeddings for all statutes
        chunk_id = 0
        for s in tqdm(statutes):  # TODO: Print progress to the terminal
            # Skip empty statutes
            if s.text.strip() == "":
                continue

            chunks = split_text_to_chunks(s.text, 8096)

            # Iterate through each chunk and embed it
            for i, c in enumerate(chunks):
                e = embed_text(c, openai_client, OPENAI_EMBED_MODEL)
                embedding = Embedding(
                    id=chunk_id,
                    statute_id=s.id,
                    text=c,
                    token_count=s.token_count,
                    model=Models.OPENAI_ADA_8K,
                    client=Clients.OPENAI,
                    embedding_vector=json.dumps(e.tolist()),
                )
                session.add(embedding)
                session.commit()
                chunk_id += 1


def generate_embedding_dump(path: str = DATA_DIR, client: str = Clients.OPENAI) -> None:
    """
    Loads the generated embeddings from the database, packs them into a numpy
    matrix and saves them to a file.

    :path: The directory to save the embeddings to.
    """
    engine = create_engine(SQL_ENGINE_PATH)
    with Session(engine) as session:
        embeddings = session.exec(
            select(Embedding).where(Embedding.client == client)
        ).all()

        # Generate a list of embeddings
        embedding_list = []
        for e in embeddings:
            embedding_list.append(json.loads(e.embedding_vector))

        query = """What is the safe harbour rule for underpayment of taxes?"""

        e1 = embed_text(query, openai_client, OPENAI_EMBED_MODEL)
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
        print(indices[:100])
        print(similarities[indices[:100]])
        print(embeddings[indices[0]].text)

        # Save the matrix to a file
        np.save(f"{path}/together_embeddings.npy", embedding_matrix)


if __name__ == "__main__":
    #load_into_db()
    #embed_statutes_together_API()
    embed_statutes_openai_API()
    generate_embedding_dump()
