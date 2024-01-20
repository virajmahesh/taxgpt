"""
    Utility functions to load the tax code into a database, and generate
    embeddings for each section.

    Author: Viraj Mahesh (virajmahesh@gmail.com)
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from openai import OpenAI
from sqlalchemy import Engine
from sqlalchemy.orm import relationship
from sqlmodel import Field, Session, SQLModel, create_engine, select
from tqdm import tqdm

from config import *
from embed import *


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

    @property
    def statute(self) -> Statute:
        """
        Get the statute that this embedding belongs to.
        """
        engine = create_engine(SQL_ENGINE_PATH)
        with Session(engine) as session:
            return session.exec(
                select(Statute).where(Statute.id == self.statute_id)
            ).one()


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


def embed_statute(s, model: EmbeddingModel, engine: Engine, **kwargs):
    """
    Embed :s: using the specified model, and store it in the database.

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
            e = model.embed_text(c, **kwargs)
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


def embed_statutes(
    model: EmbeddingModel, offset: int = 0, max_workers: int = 10, **kwargs
) -> None:
    """
    Embed all statutes using the OpenAI API. The statutes are split into chunks
    of 8096 tokens, and each chunk is embedded separately.

    :model: The model to use to generate the embeddings.

    :offset: The offset to start at. This is useful if the embedding process
    is interrupted, and needs to be restarted from a specific point.

    :max_workers: The max number of threads to use.
    """

    engine = create_engine(SQL_ENGINE_PATH)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        statutes = session.exec(select(Statute).offset(offset)).all()

        # Generate embeddings for all statutes
        # Each thread is given one full statute to deal with.
        with ThreadPoolExecutor(max_workers=max_workers) as t:
            with tqdm(total=len(statutes)) as progress:
                # Pass the keyword arguments to the embed function
                embed_func = lambda s: embed_statute(s, model, engine, **kwargs)
                futures = [t.submit(embed_func, s) for s in statutes]
                for _ in as_completed(futures):
                    progress.update(1)


def delete_failed_embedding(model: EmbeddingModel) -> None:
    """
    Delete embeddings that failed the first time.
    """
    engine = create_engine(SQL_ENGINE_PATH)
    with Session(engine) as session:
        chunks = session.exec(
            select(Embedding)
            .where(Embedding.model == model.name)
            .where(Embedding.provider == model.provider)
            .where(Embedding.embedding_vector == "[]")
        ).all()
        [session.delete(c) for c in chunks]
        session.commit()


def delete_all_embeddings(model: EmbeddingModel) -> None:
    """
    Delete all embeddings for a model.
    """
    engine = create_engine(SQL_ENGINE_PATH)
    with Session(engine) as session:
        chunks = session.exec(
            select(Embedding)
            .where(Embedding.model == model.name)
            .where(Embedding.provider == model.provider)
        ).all()
        [session.delete(c) for c in chunks]
        session.commit()


def generate_embedding_dump(
    model: EmbeddingModel, path: str = DATA_DIR, **kwargs
) -> None:
    """
    Loads the generated embeddings from the database, packs them into a numpy
    matrix and saves them to a file.

    :model: The model to use to generate the embeddings.
    :name: The file name to use
    :path: The directory to save the embeddings to.
    """
    engine = create_engine(SQL_ENGINE_PATH)
    with Session(engine) as session:
        embeddings = session.exec(
            select(Embedding)
            .where(Embedding.provider == model.provider)
            .where(Embedding.model == model.name)
            .order_by(Embedding.id)
        ).all()

        # Generate a list of embeddings
        embedding_list = []
        for e in embeddings:
            embedding_list.append(json.loads(e.embedding_vector))

        embedding_matrix = np.asarray(embedding_list).T
        embedding_matrix /= np.linalg.norm(embedding_matrix, axis=0)

        # Save the matrix to a file
        np.save(f"{path}/{model.file_safe_name()}.npy", embedding_matrix)


if __name__ == "__main__":
    # load_into_db()
    # embed_statutes(model=CohereV3English, input_type=InputTypes.SEARCH_DOCUMENT)
    generate_embedding_dump(model=Together32K)
    generate_embedding_dump(model=Together8K)
    generate_embedding_dump(model=OpenAIADA8K)
    generate_embedding_dump(model=Cohere)
    generate_embedding_dump(model=UAELarge)
    generate_embedding_dump(model=BAAILarge)
