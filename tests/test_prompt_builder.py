from backend.rag.prompt_builder import build_prompt, build_rewrite_prompt

chunks = [{"text": "Pandas is a data analysis library."}]


def test_build_prompt_without_history_matches_original_format():
    prompt = build_prompt("What is pandas?", chunks)

    assert "Conversation history" not in prompt
    assert "Pandas is a data analysis library." in prompt
    assert "What is pandas?" in prompt


def test_build_prompt_with_history_includes_history_section():
    history = [
        {"role": "user", "content": "What is numpy?"},
        {"role": "assistant", "content": "Numpy is a numerical computing library."},
    ]

    prompt = build_prompt("What about pandas?", chunks, history=history)

    assert "Conversation history" in prompt
    assert "user: What is numpy?" in prompt
    assert "assistant: Numpy is a numerical computing library." in prompt
    assert prompt.index("Conversation history") < prompt.index("Context :")


def test_build_prompt_with_empty_history_list_omits_history_section():
    assert build_prompt("q", chunks, history=[]) == build_prompt("q", chunks, history=None)
    assert "Conversation history" not in build_prompt("q", chunks, history=[])


def test_build_rewrite_prompt_includes_history_and_question():
    history = [
        {"role": "user", "content": "What is the termination clause?"},
        {"role": "assistant", "content": "It requires 30 days notice."},
    ]

    prompt = build_rewrite_prompt(history, "What about the penalty for that?")

    assert "What is the termination clause?" in prompt
    assert "It requires 30 days notice." in prompt
    assert "What about the penalty for that?" in prompt
    assert prompt.strip().endswith("Standalone question:")
