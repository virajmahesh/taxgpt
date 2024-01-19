"""
    A FastAPI server to access statutes and embeddings.
"""
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from sqlmodel import Session, create_engine, select

from config import SQL_ENGINE_PATH
from statute import Statute

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


def get_statute_by_query(
    query: str, hyde: bool, prune: bool, count: int
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

    :return: A JSON response containing the statutes' text, title and section number
    """
    pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
