"""Request and response models of the GroundedGuard API."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

RESPONSE_FIELD = Field(min_length=1, description="The text to check. Sent to the model exactly as given (not stripped).")


class QARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["qa"]
    question: str = Field(description="The user question.")
    passages: str = Field(description="Retrieved passages the answer should be grounded in.")
    response: str = RESPONSE_FIELD


class SummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["summary"]
    document: str = Field(description="The source document.")
    response: str = RESPONSE_FIELD


class Data2TxtRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["data2txt"]
    data: dict[str, Any] = Field(description="A structured record (JSON object), serialized as `key: value` lines.")
    response: str = RESPONSE_FIELD


DetectRequest = Annotated[Union[QARequest, SummaryRequest, Data2TxtRequest], Field(discriminator="task")]


class Span(BaseModel):
    start: int = Field(description="Start char offset into the original response.")
    end: int = Field(description="End char offset (exclusive); response[start:end] == text.")
    text: str
    score: float = Field(description="Max token probability inside the span (uncalibrated).")


class DetectResponse(BaseModel):
    has_hallucination: bool = Field(description="True if at least one span is flagged.")
    score: float = Field(description="Max token probability in the response: uncalibrated; compare with threshold.")
    threshold: float = Field(description="Token probability >= threshold is flagged (frozen on dev).")
    spans: list[Span]
    windowed: bool = Field(description="True if the context was split into sliding windows.")
    n_windows: int
    model: str
    latency_ms: float = Field(description="Model time for this request (batch: batch time / batch size).")
    request_id: str


class BatchResponse(BaseModel):
    request_id: str
    results: list[DetectResponse]


class Health(BaseModel):
    status: str
    model: str
    precision: str
    device: str
    threshold: float | None
    version: str


_PASSAGES = ("passage 1: The Riverside Public Library opened in 1998 in the former town hall on Elm Street. "
             "Its collection holds about 120,000 books.\n\npassage 2: Library cards are free for residents, and "
             "members can borrow up to 15 items at a time for three weeks.")
# Same text as examples/demo_context.txt (tests/test_api.py checks they stay identical); the image
# does not ship examples/.
DEMO_PASSAGES = 'passage 1: The Riverside Public Library opened in 1998 in the former town hall on Elm Street. Its collection holds about 120,000 books, including a local-history archive of maps and newspapers going back to 1870.\n\npassage 2: The library is open Monday to Saturday from 9 a.m. to 8 p.m. Library cards are free for residents, and members can borrow up to 15 items at a time for three weeks.\n'
DEMO_QUESTION = 'When did the Riverside Public Library open, and how many items can members borrow?'
EXAMPLES = {
    "qa": {"summary": "QA: question + passages",
           "value": {"task": "qa", "question": "How many items can members borrow?", "passages": _PASSAGES,
                     "response": "Members can borrow up to 25 items at a time for three weeks."}},
    "qa_supported": {"summary": "QA: a supported answer (no span expected)",
                     "value": {"task": "qa", "question": DEMO_QUESTION, "passages": DEMO_PASSAGES,
                               "response": "Library cards are free for residents, and members can borrow up to "
                                           "15 items at a time."}},
    "summary": {"summary": "Summary: document",
                "value": {"task": "summary",
                          "document": "The city council voted 7-2 on Tuesday to extend the bike lane on Main Street "
                                      "by two miles. Construction is planned to start in May and cost $1.2 million.",
                          "response": "The council unanimously approved a two-mile bike lane extension that will "
                                      "cost $1.2 million."}},
    "data2txt": {"summary": "Data-to-text: structured record",
                 "value": {"task": "data2txt",
                           "data": {"name": "Blue Door Cafe", "city": "Tucson", "categories": "Cafes, Breakfast",
                                    "hours": {"Monday": "7:00-14:00", "Sunday": "8:00-13:00"},
                                    "attributes": {"WiFi": "free", "OutdoorSeating": True}},
                           "response": "Blue Door Cafe in Tucson serves breakfast, has free WiFi and outdoor seating, "
                                       "and is open late on weekends."}},
}
BATCH_EXAMPLES = {"mixed": {"summary": "One request per task",
                            "value": [EXAMPLES[k]["value"] for k in ("qa", "summary", "data2txt")]}}
