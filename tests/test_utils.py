"""Tests for tokenizer/prompt utility fallbacks."""
from __future__ import annotations

from src.utils import format_prompt


class NoChatTemplateTokenizer:
    chat_template = None

    def apply_chat_template(self, *_args, **_kwargs):
        raise AssertionError("plain fallback should not call apply_chat_template")


class ChatTemplateTokenizer:
    chat_template = "template"

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"CHAT::{conversation[0]['content']}"


def test_format_prompt_falls_back_to_plain_instruction_without_chat_template() -> None:
    assert format_prompt("Explain gravity.", None, NoChatTemplateTokenizer()) == "Explain gravity."


def test_format_prompt_uses_tokenizer_chat_template_when_available() -> None:
    assert format_prompt("Explain gravity.", None, ChatTemplateTokenizer()) == "CHAT::Explain gravity."

