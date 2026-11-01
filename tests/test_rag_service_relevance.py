from backend.rag.rag_service import RagService, NO_ANSWER_MESSAGE


class FakeRetrievalService:
    def __init__(self, chunks):
        # chunks: list of (id, text, document_id, page_number, distance)
        self._chunks = chunks

    def retrieve(self, question, top_k=5, document_ids=None):
        return {
            "ids": [[c[0] for c in self._chunks]],
            "documents": [[c[1] for c in self._chunks]],
            "metadatas": [[{"document_id": c[2], "page_number": c[3]} for c in self._chunks]],
            "distances": [[c[4] for c in self._chunks]],
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


def test_answer_calls_llm_when_all_chunks_within_threshold():
    retrieval = FakeRetrievalService([("c1", "relevant text", "doc1", 1, 0.1)])
    llm = FakeLLMService(["final answer"])
    rag_service = RagService(retrieval_service=retrieval, llm_service=llm)

    result = rag_service.answer("What is pandas?")

    assert len(llm.prompts) == 1
    assert result["answer"] == "final answer"
    assert len(result["sources"]) == 1


def test_answer_short_circuits_when_all_chunks_beyond_threshold():
    retrieval = FakeRetrievalService([("c1", "irrelevant text", "doc1", 1, 1.9)])
    llm = FakeLLMService([])
    rag_service = RagService(retrieval_service=retrieval, llm_service=llm)

    result = rag_service.answer("What is pandas?")

    assert llm.prompts == []
    assert result["answer"] == NO_ANSWER_MESSAGE
    assert result["sources"] == []
    assert result["rewritten_question"] == "What is pandas?"


def test_answer_short_circuits_when_no_chunks_retrieved():
    retrieval = FakeRetrievalService([])
    llm = FakeLLMService([])
    rag_service = RagService(retrieval_service=retrieval, llm_service=llm)

    result = rag_service.answer("What is pandas?")

    assert llm.prompts == []
    assert result["answer"] == NO_ANSWER_MESSAGE
    assert result["sources"] == []


def test_answer_passes_only_relevant_chunks_to_prompt_with_mixed_results():
    retrieval = FakeRetrievalService([
        ("c1", "RELEVANT_CHUNK_TEXT", "doc1", 1, 0.1),
        ("c2", "IRRELEVANT_CHUNK_TEXT", "doc1", 2, 1.9),
    ])
    llm = FakeLLMService(["final answer"])
    rag_service = RagService(retrieval_service=retrieval, llm_service=llm)

    result = rag_service.answer("What is pandas?")

    final_prompt = llm.prompts[0]
    assert "RELEVANT_CHUNK_TEXT" in final_prompt
    assert "IRRELEVANT_CHUNK_TEXT" not in final_prompt
    assert len(result["sources"]) == 1
    assert result["sources"][0]["chunk_id"] == "c1"
