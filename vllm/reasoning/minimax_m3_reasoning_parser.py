# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class MiniMaxM3ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for MiniMax M3 explicit thinking blocks.

    MiniMax M3 emits reasoning as:

        <mm:think>reasoning text</mm:think>assistant content

    The M3 tokenizer exposes both markers as complete vocabulary tokens. The
    chat template may also prefill the start marker when
    ``thinking_mode="enabled"``, so generated text can begin directly inside a
    reasoning block without emitting ``<mm:think>`` again.
    """

    @property
    def start_token(self) -> str:
        return "<mm:think>"

    @property
    def end_token(self) -> str:
        return "</mm:think>"

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self._thinking_mode = chat_kwargs.get("thinking_mode")
        self._initial_in_reasoning = self._thinking_mode == "enabled"
        self._at_response_start = True
        self._stream_buffer = ""
        self._stream_in_reasoning = self._initial_in_reasoning
        self._stream_done_reasoning = False
        self._text_end_content_token_ids: list[int] | None = None

    @staticmethod
    def _pending_marker_prefix_len(text: str, marker: str) -> int:
        max_len = min(len(text), len(marker) - 1)
        for length in range(max_len, 0, -1):
            if marker.startswith(text[-length:]):
                return length
        return 0

    def _encode_content_text(self, content: str) -> list[int]:
        return self.model_tokenizer.encode(content, add_special_tokens=False)

    @staticmethod
    def _merge_delta(
        delta: DeltaMessage | None, extra: DeltaMessage | None
    ) -> DeltaMessage | None:
        if extra is None:
            return delta
        if delta is None:
            return extra
        if extra.reasoning:
            delta.reasoning = (delta.reasoning or "") + extra.reasoning
        if extra.content:
            delta.content = (delta.content or "") + extra.content
        return delta

    @staticmethod
    def _make_delta(
        reasoning: str | None = None, content: str | None = None
    ) -> DeltaMessage | None:
        if not reasoning and not content:
            return None
        return DeltaMessage(reasoning=reasoning or None, content=content or None)

    def _consume_reasoning_buffer(self) -> DeltaMessage | None:
        if self.end_token in self._stream_buffer:
            reasoning, _, content = self._stream_buffer.partition(self.end_token)
            self._stream_buffer = ""
            self._stream_in_reasoning = False
            self._stream_done_reasoning = True
            self._text_end_content_token_ids = self._encode_content_text(content)
            return self._make_delta(reasoning=reasoning, content=content)

        pending = self._pending_marker_prefix_len(self._stream_buffer, self.end_token)
        reasoning = self._stream_buffer[: len(self._stream_buffer) - pending]
        self._stream_buffer = self._stream_buffer[len(reasoning) :]
        return self._make_delta(reasoning=reasoning)

    def _consume_pre_reasoning_buffer(self) -> DeltaMessage | None:
        if self._at_response_start:
            # Apply the leading-closer tolerance once. Later unmatched closers
            # stay visible as content.
            if (
                self.end_token.startswith(self._stream_buffer)
                and self._stream_buffer != self.end_token
            ):
                return None
            self._at_response_start = False
            if self._stream_buffer.startswith(self.end_token):
                content = self._stream_buffer[len(self.end_token) :]
                self._stream_buffer = ""
                self._stream_done_reasoning = True
                self._text_end_content_token_ids = self._encode_content_text(content)
                return self._make_delta(content=content)

        if self.start_token in self._stream_buffer:
            content, _, reasoning = self._stream_buffer.partition(self.start_token)
            self._stream_buffer = reasoning
            self._stream_in_reasoning = True
            delta = self._make_delta(content=content)
            delta = self._merge_delta(delta, self._consume_reasoning_buffer())
            if self._stream_done_reasoning and delta and delta.content:
                self._text_end_content_token_ids = self._encode_content_text(
                    delta.content
                )
            return delta

        pending = self._pending_marker_prefix_len(self._stream_buffer, self.start_token)
        content = self._stream_buffer[: len(self._stream_buffer) - pending]
        self._stream_buffer = self._stream_buffer[len(content) :]
        if content:
            self._stream_done_reasoning = True
            return self._make_delta(content=content)
        return None

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        # MiniMax M3 can start a response with a stray closer. Drop that first
        # token only; later unmatched closers stay visible as content.
        if not self._initial_in_reasoning and model_output.startswith(self.end_token):
            content = model_output[len(self.end_token) :]
            return None, content or None

        if self._initial_in_reasoning and self.start_token not in model_output:
            reasoning, end, content = model_output.partition(self.end_token)
            if not end:
                return model_output, None
            return reasoning, content or None

        if self.start_token not in model_output:
            return None, model_output

        content_before, _, after_start = model_output.partition(self.start_token)
        reasoning, end, content_after = after_start.partition(self.end_token)
        if not end:
            return reasoning, content_before or None

        return reasoning, (content_before + content_after) or None

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        delta_ids = tuple(delta_ids)
        if self._stream_done_reasoning:
            return True
        if self.end_token_id in delta_ids:
            self._stream_done_reasoning = True
            return True
        if self.end_token_id in input_ids:
            self._stream_done_reasoning = True
            return True
        if self._stream_in_reasoning:
            return False
        if self._stream_buffer:
            return False
        if self._initial_in_reasoning:
            return False
        if self.start_token_id not in input_ids:
            return bool(input_ids)
        return False

    def is_reasoning_end_for_prompt(self, input_ids: Sequence[int]) -> bool:
        if self._thinking_mode == "disabled":
            return True
        if self._thinking_mode == "enabled":
            return False
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._text_end_content_token_ids is not None:
            token_ids = self._text_end_content_token_ids
            self._text_end_content_token_ids = None
            return token_ids

        if self.end_token_id in input_ids:
            end_index = len(input_ids) - 1 - input_ids[::-1].index(self.end_token_id)
            return input_ids[end_index + 1 :]

        if self._initial_in_reasoning and self.start_token_id not in input_ids:
            return []

        if self.start_token_id not in input_ids:
            return input_ids
        return []

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if not delta_text:
            return None

        self._stream_buffer += delta_text

        if self._stream_done_reasoning:
            content = self._stream_buffer
            self._stream_buffer = ""
            return self._make_delta(content=content)

        if self._stream_in_reasoning:
            return self._consume_reasoning_buffer()

        return self._consume_pre_reasoning_buffer()

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        if not self._initial_in_reasoning:
            return super().count_reasoning_tokens(token_ids)

        count = 0
        depth = 1
        for token_id in token_ids:
            if token_id == self.start_token_id:
                depth += 1
                continue
            if token_id == self.end_token_id:
                if depth > 0:
                    depth -= 1
                continue
            if depth > 0:
                count += 1
        return count
