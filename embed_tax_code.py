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


# Config
DATA_DIR = "data"
TAX_CODE_XHTML_PATH = f"{DATA_DIR}/title26.htm"
EMBED_MODEL = ""  # Set this explicitly so that it doesn't change automatically
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


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
                f.write(get_text_from_html(h))
                f.write("\n")
                f.write(get_text_from_html(s))
                f.close()


if __name__ == "__main__":
    split_tax_code()
