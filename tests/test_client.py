"""Unit tests for lmsa.client.LMStudio.see() -- the vision entry point.

Fakes at the `openai.OpenAI` client boundary (same spot `chat`/`extract`
would be tested at): a stand-in `.chat.completions.create(**kwargs)` that
records the call and returns a canned response shaped like the real SDK's.
"""

from __future__ import annotations

from dataclasses import dataclass

from lmsa.client import LMStudio, _sniff_image_mime
from lmsa.config import Settings

PNG_MAGIC = b"\x89PNG\r\n\x1a\nrest-of-a-fake-png"
JPEG_MAGIC = b"\xff\xd8\xffrest-of-a-fake-jpeg"


def _cfg(**overrides) -> Settings:
    base = dict(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        chat_model="chat-model",
        embed_model="embed-model",
        vision_model="vision-model",
        output_dir="/tmp/out",
        timeout=300.0,
        imap_host="imap.gmail.com",
        imap_port=993,
        imap_user="",
    )
    base.update(overrides)
    return Settings(**base)


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list
    usage: object = None


class _FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(choices=[_Choice(_Message(self.content))])


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeAPI:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)
        self.chat = _FakeChat(self.completions)


def _lm_with_fake(content: str) -> tuple[LMStudio, _FakeCompletions]:
    lm = LMStudio(_cfg())
    fake = _FakeAPI(content)
    lm.api = fake
    return lm, fake.completions


def test_sniff_image_mime_png_jpeg_gif_and_default():
    assert _sniff_image_mime(PNG_MAGIC) == "image/png"
    assert _sniff_image_mime(JPEG_MAGIC) == "image/jpeg"
    assert _sniff_image_mime(b"GIF89a...") == "image/gif"
    assert _sniff_image_mime(b"not a real image") == "image/png"


def test_see_without_schema_returns_free_text_like_chat():
    lm, completions = _lm_with_fake("<think>plan</think>a receipt for $9.99")
    result = lm.see([PNG_MAGIC], "what is this?")

    assert result == "a receipt for $9.99"  # strip_think applied, same as chat()
    kwargs = completions.calls[0]
    assert kwargs["model"] == "vision-model"  # settings().vision_model default
    assert kwargs["temperature"] == 0.2
    assert "response_format" not in kwargs


def test_see_with_schema_returns_constrained_json_like_extract():
    lm, completions = _lm_with_fake('{"is_receipt": true, "amount": "9.99"}')
    schema = {"type": "object", "properties": {"is_receipt": {"type": "boolean"}}}

    result = lm.see([PNG_MAGIC], "extract fields", schema, name="receipt")

    assert result == {"is_receipt": True, "amount": "9.99"}
    kwargs = completions.calls[0]
    assert kwargs["temperature"] == 0
    assert kwargs["response_format"]["json_schema"]["name"] == "receipt"
    assert kwargs["response_format"]["json_schema"]["schema"] is schema


def test_see_builds_one_image_url_part_per_image_with_sniffed_mime():
    lm, completions = _lm_with_fake("ok")
    lm.see([PNG_MAGIC, JPEG_MAGIC], "look at both")

    content = completions.calls[0]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "look at both"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_see_respects_system_and_model_overrides():
    lm, completions = _lm_with_fake("ok")
    lm.see([PNG_MAGIC], "p", system="be careful", model="other-vision-model")

    kwargs = completions.calls[0]
    assert kwargs["model"] == "other-vision-model"
    assert kwargs["messages"][0] == {"role": "system", "content": "be careful"}
