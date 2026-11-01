def build_prompt(question,chunks,history=None):
    context = "\n\n".join(chunk["text"] for chunk in chunks)
    history_block = ""
    if history:
        history_lines = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
        history_block = f"""Conversation history (for context only; the Context section below is the source of truth for facts):
{history_lines}

"""
    return f"""
                You are a document question-answering assistant.
                Answer the user's question using only the provided context.
                If the answer cannot be found in the context, say that you could not find the answer in the uploaded documents do not invent facts.
                {history_block}Context : {context}
                Question :{question}
                Answer:
                """


def build_rewrite_prompt(history,question):
    history_lines = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    return f"""Given the conversation history and a follow-up question, rewrite the follow-up question into a standalone question that includes all necessary context from the history. Resolve pronouns and vague references (e.g. "it", "that", "this") using the history.

If the follow-up question is already standalone and does not depend on the history, return it unchanged.

Only output the rewritten standalone question. Do not add commentary, labels, or quotation marks.

Conversation history:
{history_lines}

Follow-up question: {question}

Standalone question:"""
