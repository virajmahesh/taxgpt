import os
import tiktoken
from openai import OpenAI
import together

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