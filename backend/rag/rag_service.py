from backend.rag.prompt_builder import build_prompt, build_rewrite_prompt
from backend.rag.extract_chunks_from_retrieve import extract_chunks
from backend.rag.relevance import filter_relevant_chunks

NO_ANSWER_MESSAGE = "I could not find the answer to that in the uploaded documents."


class RagService:
    def __init__(self,retrieval_service,llm_service):
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service

    def answer(self,question,history=None,document_ids=None):
        history = history or []

        retrieval_question = question
        if history:
            retrieval_question = self._rewrite_question(question,history)

        results = self.retrieval_service.retrieve(retrieval_question,top_k=5,document_ids=document_ids)
        chunks = extract_chunks(results)
        relevant_chunks = filter_relevant_chunks(chunks)

        if not relevant_chunks:
            return {
                "answer": NO_ANSWER_MESSAGE,
                "sources": [],
                "rewritten_question": retrieval_question,
            }

        prompt = build_prompt(question,relevant_chunks,history=history)
        answer = self.llm_service.generate(prompt)

        return {
            "answer" : answer,
            "sources": relevant_chunks,
            "rewritten_question": retrieval_question,
        }

    def _rewrite_question(self,question,history):
        prompt = build_rewrite_prompt(history,question)
        try:
            rewritten = self.llm_service.generate(prompt)
        except Exception:
            return question
        rewritten = (rewritten or "").strip()
        return rewritten if rewritten else question
