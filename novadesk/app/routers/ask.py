"""POST /ask: thin on purpose. Validate the request, call the pipeline, return its result."""

from typing import Annotated

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, StringConstraints

from app.rag.pipeline import AnswerResult, ProviderUnavailableError, answer_question

router = APIRouter()

# Shown to the employee when Gemini is over quota or unreachable.
_UNAVAILABLE_MESSAGE = "השירות לא זמין כרגע, נסו שוב בעוד דקה"


class AskRequest(BaseModel):
    # Strip spaces from both ends, then require at least 1 character:
    # "" and "   " are both rejected with 422 before our code even runs.
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


# A plain `def` (not `async def`): the pipeline makes blocking network calls,
# so FastAPI runs this in a worker thread and the server stays responsive.
@router.post(
    "/ask",
    response_model=AnswerResult,
    responses={503: {"description": "The AI provider is temporarily unavailable"}},
)
def ask(request: AskRequest) -> AnswerResult:
    try:
        return answer_question(request.question)
    except ProviderUnavailableError:
        raise HTTPException(status_code=503, detail=_UNAVAILABLE_MESSAGE)
