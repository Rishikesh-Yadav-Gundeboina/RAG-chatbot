from backend.rag.rag_service import RagService


class FakeRetrievalService:
    def __init__(self):
        self.received_questions = []

    def retrieve(self, question, top_k=5, document_ids=None):
        self.received_questions.append(question)
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


def test_answer_without_history_skips_rewrite_and_uses_raw_question():
    retrieval = FakeRetrievalService()
    llm = FakeLLMService(["final answer"])
    rag_service = RagService(retrieval_service=retrieval, llm_service=llm)

    result = rag_service.answer("What is pandas?")

    assert len(llm.prompts) == 1
    assert retrieval.received_questions == ["What is pandas?"]
    assert result["rewritten_question"] == "What is pandas?"
    assert result["answer"] == "final answer"


def test_answer_with_history_rewrites_question_for_retrieval_only():
    retrieval = FakeRetrievalService()
    llm = FakeLLMService(["What is the penalty for late termination?", "final answer"])
    rag_service = RagService(retrieval_service=retrieval, llm_service=llm)
    history = [
        {"role": "user", "content": "What is the termination clause?"},
        {"role": "assistant", "content": "It requires 30 days notice."},
    ]

    result = rag_service.answer("What about the penalty for that?", history=history)

    assert retrieval.received_questions == ["What is the penalty for late termination?"]
    final_prompt = llm.prompts[1]
    assert "What about the penalty for that?" in final_prompt
    assert "What is the termination clause?" in final_prompt
    assert result["rewritten_question"] == "What is the penalty for late termination?"


def test_answer_falls_back_to_raw_question_when_rewrite_fails():
    retrieval = FakeRetrievalService()
    llm = FakeLLMService([RuntimeError("api down"), "final answer"])
    rag_service = RagService(retrieval_service=retrieval, llm_service=llm)
    history = [{"role": "user", "content": "prior question"}]

    result = rag_service.answer("follow up", history=history)

    assert retrieval.received_questions == ["follow up"]
    assert result["rewritten_question"] == "follow up"


def test_answer_falls_back_when_rewrite_returns_empty_string():
    retrieval = FakeRetrievalService()
    llm = FakeLLMService(["   ", "final answer"])
    rag_service = RagService(retrieval_service=retrieval, llm_service=llm)
    history = [{"role": "user", "content": "prior question"}]

    result = rag_service.answer("follow up", history=history)

    assert retrieval.received_questions == ["follow up"]
    assert result["rewritten_question"] == "follow up"
