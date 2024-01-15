"""
    Constants used throughout the project.

    Author: Viraj Mahesh (virajmahesh@gmail.com)
"""

import os
import together
import tiktoken
from openai import OpenAI

# File paths
DATA_DIR = "data"
TAX_CODE_XHTML_PATH = f"{DATA_DIR}/title26.htm"

# OpenAI
OPENAI_EMBED_MODEL = "text-embedding-ada-002"  # Set this explicitly so that it doesn't change automatically
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Together.AI
TOGETHER_EMBED_MODEL = "togethercomputer/m2-bert-80M-32k-retrieval"
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]

# SQL Enginge
SQL_ENGINE = "sqlite"
DB_PATH = "statute.db"
SQL_ENGINE_PATH = f"{SQL_ENGINE}:///{DB_PATH}"
