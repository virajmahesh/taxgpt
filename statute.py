"""
    This file contains utility functions to generate embeddings for the US tax
    code. We assume the tax code has already been downloaded in XHTML format 
    from https://uscode.house.gov/download/download.shtml.

    Author: Viraj Mahesh (virajmahesh@gmail.com)
"""

import os
import re
from openai import OpenAI
from bs4 import BeautifulSoup
from sqlmodel import SQLModel, Field, create_engine, Session


# Config
DATA_DIR = "data"
TAX_CODE_XHTML_PATH = f"{DATA_DIR}/title26.htm"
EMBED_MODEL = ""  # Set this explicitly so that it doesn't change automatically
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


class Statute(SQLModel, table=True):
    """
    A model for a statute in the US tax code.
    """
    id: int = Field(default=None, primary_key=True)
    section: str = Field(default=None)
    title: str = Field(default=None)
    text: str = Field(default=None)


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
    Load the statute text at :path: into a SQLite database.
    """
    # Create the database and table
    engine = create_engine('sqlite:///statute.db')
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

                # The first line of the file looks like §{section_number} {section_title}
                result = re.search(r"§(\S+\.\S+) (.*)", first_line)
                if result is None:
                    result = re.search(r"§(\S+)\. (.*)", first_line)
                section, title = result.groups()
                title = title.strip()

                # Read the rest of the file, and remove any leading or trailing whitespace
                text = statute_file.read().strip()
                statute = Statute(section=section, title=title, text=text)

                # Add the statute to the database
                session.add(statute)
        session.commit() # Commit the changes
    

if __name__ == "__main__":
    load_into_db()
