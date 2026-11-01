import httpx
import pytest

from backend.rag.llm_service import LLMService, LLMTimeoutError, LLM_TIMEOUT_SECONDS
from backend.rag.rag_service import RagService


class FakeRetrievalService:
    def retrieve(self, question, top_k=5, document_ids=None):
        return {
            "ids": [["c1"]],
            "documents": [["chunk text"]],
            "metadatas": [[{"document_id": "doc1", "page_number": 1}]],
            "distances": [[0.1]],
        }


class FakeLLMService:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeModels:
    def __init__(self, exc):
        self._exc = exc

    def generate_content(self, model, contents):
        raise self._exc


class FakeClient:
    def __init__(self, api_key=None, http_options=None):
        self.http_options = http_options
        self.models = FakeModels(httpx.ReadTimeout("timed out"))


def test_generate_raises_llm_timeout_error_on_httpx_timeout(monkeypatch):
    monkeypatch.setattr("backend.rag.llm_service.genai.Client", FakeClient)
    llm_service = LLMService()

    with pytest.raises(LLMTimeoutError):
        llm_service.generate("What is pandas?")


def test_generate_configures_client_with_timeout_in_milliseconds(monkeypatch):
    captured = {}

    class CapturingClient(FakeClient):
        def __init__(self, api_key=None, http_options=None):
            super().__init__(api_key=api_key, http_options=http_options)
            captured["http_options"] = http_options

    monkeypatch.setattr("backend.rag.llm_service.genai.Client", CapturingClient)
    llm_service = LLMService()

    with pytest.raises(LLMTimeoutError):
        llm_service.generate("What is pandas?")

    assert captured["http_options"].timeout == LLM_TIMEOUT_SECONDS * 1000


def test_rewrite_falls_back_to_raw_question_on_llm_timeout():
    retrieval = FakeRetrievalService()
    llm = FakeLLMService([LLMTimeoutError("timed out"), "final answer"])
    rag_service = RagService(retrieval_service=retrieval, llm_service=llm)
    history = [{"role": "user", "content": "prior question"}]

    result = rag_service.answer("follow up", history=history)

    assert result["rewritten_question"] == "follow up"
    assert result["answer"] == "final answer"


def test_answer_propagates_llm_timeout_error_from_main_generation():
    retrieval = FakeRetrievalService()
    llm = FakeLLMService([LLMTimeoutError("timed out")])
    rag_service = RagService(retrieval_service=retrieval, llm_service=llm)

    with pytest.raises(LLMTimeoutError):
        rag_service.answer("What is pandas?")
