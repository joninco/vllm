# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import string
from collections.abc import Sequence

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser import ParserManager
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser

pytestmark = pytest.mark.skip_global_cleanup


class MiniMaxM3Tokenizer:
    """Small tokenizer with MiniMax M3 reasoning tags as special tokens."""

    special_tokens = ("<mm:think>", "</mm:think>")

    def __init__(self):
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        for token in self.special_tokens:
            self._add_token(token)
        for char in string.printable:
            self._add_token(char)

    def _add_token(self, token: str) -> int:
        token_id = self._token_to_id.get(token)
        if token_id is None:
            token_id = len(self._token_to_id) + 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return token_id

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        return [self._add_token(token) for token in self.tokenize(text)]

    def decode(
        self, ids: Sequence[int] | int, skip_special_tokens: bool = False
    ) -> str:
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self._id_to_token[token_id] for token_id in ids)

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        pos = 0
        while pos < len(text):
            for special_token in self.special_tokens:
                if text.startswith(special_token, pos):
                    tokens.append(special_token)
                    pos += len(special_token)
                    break
            else:
                tokens.append(text[pos])
                pos += 1
        return tokens

    def convert_ids_to_tokens(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return [self._id_to_token[token_id] for token_id in ids]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self._add_token(tokens)
        return [self._add_token(token) for token in tokens]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)


class MiniMaxM3TextSplitTokenizer(MiniMaxM3Tokenizer):
    """Tokenizer that exposes M3 tags in vocab but emits them as text tokens."""

    def tokenize(self, text: str) -> list[str]:
        return list(text)


def make_parser(
    chat_template_kwargs: dict[str, str] | None = None,
    tokenizer_cls: type[MiniMaxM3Tokenizer] = MiniMaxM3Tokenizer,
) -> tuple[MiniMaxM3ReasoningParser, MiniMaxM3Tokenizer]:
    tokenizer = tokenizer_cls()
    return (
        MiniMaxM3ReasoningParser(tokenizer, chat_template_kwargs=chat_template_kwargs),
        tokenizer,
    )


def run_streaming(
    parser: MiniMaxM3ReasoningParser,
    tokenizer: MiniMaxM3Tokenizer,
    chunks: list[str],
) -> tuple[str | None, str | None, list[bool]]:
    previous_text = ""
    previous_token_ids: list[int] = []
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    reasoning_end_states: list[bool] = []

    for chunk in chunks:
        delta_token_ids = tokenizer.encode(chunk, add_special_tokens=False)
        current_text = previous_text + chunk
        current_token_ids = previous_token_ids + delta_token_ids
        delta = parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
        )
        reasoning_end_states.append(
            parser.is_reasoning_end_streaming(current_token_ids, delta_token_ids)
        )

        if delta is not None:
            if delta.reasoning is not None:
                reasoning_parts.append(delta.reasoning)
            if delta.content is not None:
                content_parts.append(delta.content)

        previous_text = current_text
        previous_token_ids = current_token_ids

    return (
        "".join(reasoning_parts) or None,
        "".join(content_parts) or None,
        reasoning_end_states,
    )


def run_unified_streaming(
    parser,
    tokenizer: MiniMaxM3Tokenizer,
    chunks: list[str],
    prompt_text: str,
    request: ChatCompletionRequest | None = None,
) -> tuple[str | None, str | None]:
    request = request or ChatCompletionRequest(messages=[], model="test-model")
    prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    for idx, chunk in enumerate(chunks):
        delta = parser.parse_delta(
            delta_text=chunk,
            delta_token_ids=tokenizer.encode(chunk, add_special_tokens=False),
            request=request,
            prompt_token_ids=prompt_token_ids if idx == 0 else None,
            finished=False,
        )
        if delta is not None:
            if delta.reasoning is not None:
                reasoning_parts.append(delta.reasoning)
            if delta.content is not None:
                content_parts.append(delta.content)

    return "".join(reasoning_parts) or None, "".join(content_parts) or None


def sample_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="lookup",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
        )
    ]


def test_parser_registration():
    parser_cls = ReasoningParserManager.get_reasoning_parser("minimax_m3")

    assert parser_cls is MiniMaxM3ReasoningParser


def test_nonstreaming_extracts_explicit_reasoning_block():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning(
        "<mm:think>plan</mm:think>answer", request
    )

    assert reasoning == "plan"
    assert content == "answer"


def test_nonstreaming_without_start_tag_is_content():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("plain answer", request)

    assert reasoning is None
    assert content == "plain answer"


def test_nonstreaming_drops_leading_end_tag():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("</mm:think>answer", request)

    assert reasoning is None
    assert content == "answer"


def test_nonstreaming_non_leading_end_tag_is_content():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("XXX</mm:think>YYY", request)

    assert reasoning is None
    assert content == "XXX</mm:think>YYY"


def test_nonstreaming_enabled_mode_starts_in_reasoning():
    parser, _ = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("plan</mm:think>answer", request)

    assert reasoning == "plan"
    assert content == "answer"


def test_nonstreaming_open_reasoning_block():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("<mm:think>still thinking", request)

    assert reasoning == "still thinking"
    assert content is None


def test_streaming_reasoning_tags_are_not_returned():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["<mm:think>", "plan", "</mm:think>", "answer"],
    )

    assert reasoning == "plan"
    assert content == "answer"
    assert end_states == [False, False, True, True]


def test_streaming_boundary_can_emit_reasoning_and_content():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["<mm:think>plan</mm:think>answer"],
    )

    assert reasoning == "plan"
    assert content == "answer"
    assert end_states == [True]


def test_streaming_text_tags_can_be_split_across_chunks():
    parser, tokenizer = make_parser(tokenizer_cls=MiniMaxM3TextSplitTokenizer)

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["<mm", ":think>", "plan", "</mm", ":think>", "answer"],
    )

    assert reasoning == "plan"
    assert content == "answer"
    assert "<mm:think>" not in (reasoning or "")
    assert "<mm:think>" not in (content or "")
    assert end_states == [False, False, False, False, True, True]


def test_streaming_text_tags_can_share_boundary_chunk():
    parser, tokenizer = make_parser(tokenizer_cls=MiniMaxM3TextSplitTokenizer)

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["<mm:think>plan</mm:think>answer"],
    )

    assert reasoning == "plan"
    assert content == "answer"
    assert end_states == [True]


def test_streaming_drops_leading_end_tag():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["</mm:think>", "answer"],
    )

    assert reasoning is None
    assert content == "answer"
    assert end_states == [True, True]


def test_streaming_non_leading_end_tag_is_content():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["XXX</mm:think>YYY"],
    )

    assert reasoning is None
    assert content == "XXX</mm:think>YYY"
    assert end_states == [True]


def test_streaming_enabled_mode_starts_in_reasoning():
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["plan", "</mm:think>", "answer"],
    )

    assert reasoning == "plan"
    assert content == "answer"
    assert end_states == [False, True, True]


def test_streaming_plain_content_ends_reasoning_phase():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["plain ", "answer"],
    )

    assert reasoning is None
    assert content == "plain answer"
    assert end_states == [True, True]


def test_token_id_helpers():
    parser, tokenizer = make_parser()
    output_ids = tokenizer.encode(
        "<mm:think>abc</mm:think>def", add_special_tokens=False
    )
    open_reasoning_ids = tokenizer.encode("<mm:think>abc", add_special_tokens=False)
    content_ids = tokenizer.encode("plain", add_special_tokens=False)

    assert parser.is_reasoning_end(output_ids)
    assert not parser.is_reasoning_end(open_reasoning_ids)
    assert not parser.is_reasoning_end(content_ids)
    assert tokenizer.decode(parser.extract_content_ids(output_ids)) == "def"
    assert parser.extract_content_ids(open_reasoning_ids) == []
    assert parser.extract_content_ids(content_ids) == content_ids
    assert parser.count_reasoning_tokens(output_ids) == len(tokenizer.encode("abc"))


def test_prompt_reasoning_end_uses_m3_thinking_mode():
    parser, tokenizer = make_parser()
    prompt_ids = tokenizer.encode("</mm:think>", add_special_tokens=False)

    assert not parser.is_reasoning_end_for_prompt(prompt_ids)

    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "adaptive"})
    prompt_ids = tokenizer.encode("</mm:think>", add_special_tokens=False)
    assert not parser.is_reasoning_end_for_prompt(prompt_ids)

    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
    prompt_ids = tokenizer.encode("<mm:think>", add_special_tokens=False)
    assert not parser.is_reasoning_end_for_prompt(prompt_ids)

    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "disabled"})
    prompt_ids = tokenizer.encode("</mm:think>", add_special_tokens=False)
    assert parser.is_reasoning_end_for_prompt(prompt_ids)


def test_unified_parser_adaptive_prompt_does_not_leak_start_tag():
    tokenizer = MiniMaxM3Tokenizer()
    parser_cls = ParserManager.get_parser(reasoning_parser_name="minimax_m3")
    assert parser_cls is not None
    parser = parser_cls(
        tokenizer,
        chat_template_kwargs={"thinking_mode": "adaptive"},
    )

    reasoning, content = run_unified_streaming(
        parser,
        tokenizer,
        ["<mm:think>", "plan", "</mm:think>", "answer"],
        prompt_text="old answer</mm:think>",
    )

    assert reasoning == "plan"
    assert content == "answer"


def test_unified_parser_with_m3_tool_parser_does_not_leak_start_tag():
    pytest.importorskip("vllm._rust_tool_parser")
    tokenizer = MiniMaxM3Tokenizer()
    tools = sample_tools()
    parser_cls = ParserManager.get_parser(
        tool_parser_name="minimax_m3",
        reasoning_parser_name="minimax_m3",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    parser = parser_cls(
        tokenizer,
        tools,
        chat_template_kwargs={"thinking_mode": "adaptive"},
    )
    request = ChatCompletionRequest(
        messages=[],
        model="test-model",
        tools=tools,
    )

    reasoning, content = run_unified_streaming(
        parser,
        tokenizer,
        ["<mm:think>", "plan", "</mm:think>", "answer"],
        prompt_text="old answer</mm:think>",
        request=request,
    )

    assert reasoning == "plan"
    assert content == "answer"


def test_token_id_helpers_enabled_mode():
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
    output_ids = tokenizer.encode("abc</mm:think>def", add_special_tokens=False)
    open_reasoning_ids = tokenizer.encode("abc", add_special_tokens=False)

    assert parser.is_reasoning_end(output_ids)
    assert not parser.is_reasoning_end(open_reasoning_ids)
    assert tokenizer.decode(parser.extract_content_ids(output_ids)) == "def"
    assert parser.extract_content_ids(open_reasoning_ids) == []
    assert parser.count_reasoning_tokens(output_ids) == len(tokenizer.encode("abc"))
    assert parser.count_reasoning_tokens(open_reasoning_ids) == len(
        tokenizer.encode("abc")
    )
