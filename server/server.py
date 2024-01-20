"""
    A FastAPI server to access statutes and embeddings.
"""
from enum import Enum
from pprint import pprint

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from sqlmodel import Session, create_engine, select

from chat import ChatModel, GPT4Turbo
from config import DATA_DIR, SQL_ENGINE_PATH
from embed import EmbeddingModel, OpenAIADA8K, UAELarge
from statute import Embedding, Statute

app = FastAPI()


@app.get("/statute")
def get_statute(section: str):
    """
    Get a statute by :section:. This must match the section number
    exactly.

    #TODO: Add section suggestions if there are no exact matches.

    :section: The published section number of the statute.
    """
    engine = create_engine(SQL_ENGINE_PATH)
    with Session(engine) as session:
        statement = select(Statute).where(Statute.section == section)
        statute = session.exec(statement).first()
        if statute:
            return Response(statute.text)
        else:
            return f"Error, no statute with section number {section} found"


def get_hyde_statute(query: str, model=GPT4Turbo) -> JSONResponse:
    """
    Generates a fake statute that answers :query:
    """
    hyde_prompt = """
        Imagine that you are an aide to a Congressman in the U.S. House of Representatives.
        Your job is to update the U.S. tax code to be as clear as possible about income tax. To address this,'
        you are reviewing the questions that people have asked about income tax, and then drafting statutes
        that answer their questions. For every question the user asks, respond with a model statute that will address
        the question. It is okay if your statute is duplicative of anything that already exsists in the tax code. In fact,
        the more you draw on the existing tax code, the better.
         
        Your responses are always in plaintext. Do not use markdown or HTML formatting.
    """
    messages = [
        {"role": "system", "content": hyde_prompt},
        {"role": "user", "content": query},
    ]
    response, _ = model.chat_response(messages)
    return response


def rewrite_query(messages, model: ChatModel) -> str:
    """
    Rewrites the query in :messages: using :model: such that all relevant context is embedded in the query.
    """
    pass

def prune(
    query: str, messages: list[dict[str, str]], statutes: list[Statute], model=GPT4Turbo
) -> list[Statute]:
    """
    Prunes the statutes in :statutes: by asking :model: whether they answer :query:.
    """
    pass


def get_statute_by_query(
    query: str,
    hyde: bool,
    prune: bool,
    count: int,
    embed_model: EmbeddingModel,
    chat_model: ChatModel,
    path: str = DATA_DIR,
) -> JSONResponse:
    """
    Get :count: statutes that answer the question in :query:. The pipeline
    is as follows:

    query -> HyDE (optional) -> Embed (query or hypothetical answer) -> Retrieval
    -> Prune (optional) -> Return

    In the future, we could add re-ranking (either before or after pruning).
    This could be implemented using either the Cohere API, or using prompting
    (i.e. give an LLM the query and two answers, and ask it which one better
    answers the question)

    :param query: The question to answer.
    :param hyde: Whether to use the HyDE. HyDE generates a hypothetical answer to
    the question and finds statutes similar to it.
    :param prune: Whether to prune answers. Pruning is done by asking an LLM whether
    the statute answers the question.
    :count: The number of statutes to return.

    :embed_model: The model to use for embedding.
    :chat_model: The model to use for chat.

    :return: A JSON response containing the statutes' text, title and section number
    """
    engine = create_engine(SQL_ENGINE_PATH)
    search_query = query
    if hyde:
        search_query = get_hyde_statute(query)

    embedding_matrix = np.load(f"{path}/{embed_model.file_safe_name()}.npy")
    query_embed = embed_model.embed_text(search_query)

    similarites = query_embed.dot(embedding_matrix) / np.linalg.norm(query_embed)
    similar_idx = np.argsort(similarites)[::-1]

    with Session(engine) as session:
        results = []
        for o in similar_idx[:count]:
            e = session.exec(
                select(Embedding)
                .where(Embedding.model == embed_model.name)
                .where(Embedding.provider == embed_model.provider)
                .order_by(Embedding.id)
                .limit(1)
                .offset(o)
            ).first()
            results.append(e.statute)
    return results


def answer_question(messages: list[dict[str, str]], statutes: list[Statute]) -> str:
    answer_system_prompt = """
        You are an expert in tax law. You are having a discussion with a user who has questions about U.S. tax law.
        After the user asks the question, you research relevant statutes that might answer the question. Those statutes are provided
        in the chat history. Now, based on the statutes, respond to the user's question. Where possible, you should cite the relevant statutes
        that support your statement. Note that all statutes may not be relevant, only cite the statutes that are.

        The statutes are numbered for convienience. To cite a statute, use the following format: <response>[<statute number>]. For example, the
        conversation might look like this:

        Conversation:
        -------------
        User: What is the definition of income?
        Assistant: These are the relevant statutes:
        [1] (a) In generalExcept as provided in subsection (b), for purposes of this subtitle, the term "taxable income" means gross income minus the deductions allowed by this chapter (other than the standard deduction).
        [2] sIn the case of an insurance company subject to the tax imposed by section831—(1) Gross incomeThe term "gross income" means the sum of—(A) the combined gross amount earned during the taxable year, from investment income and from underwriting income as provided in this subsection, computed on the basis of the underwriting and investment exhibit of the annual statement approved by the National Association of Insurance Commissioners,
            (B) gain during the taxable year from the sale or other disposition of property
        
        Response:
        -------------
        Assistant: Income can mean different things depending on the context. For example, in the context of insurance companies, income is defined as the sum of the combined gross amount earned during the taxable year, from investment income and from underwriting income as provided in this subsection, computed on the basis of the underwriting and investment exhibit of the annual statement approved by the National Association of Insurance Commissioners [2].
    """
    statute_message = "\n".join(
        [f"[{i}] {statute.text}\n" + "-" * 20 for i, statute in enumerate(statutes)]
    )

    messages = (
        [{"role": "system", "content": answer_system_prompt}]
        + messages
        + [{"role": "assistant", "content": statute_message}]
    )
    response, _ = GPT4Turbo.chat_response(messages)
    return response


def respond(messages: list[dict[str, str]]) -> str:
    last_message = messages[-1]["content"]
    statutes = get_statute_by_query(last_message, True, True, 5, OpenAIADA8K, GPT4Turbo)
    answer_question(messages, statutes)
    return ""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
