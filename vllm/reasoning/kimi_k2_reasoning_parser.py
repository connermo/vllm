# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser
from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser
from vllm.tokenizers import TokenizerLike


class KimiK2ReasoningParser(ReasoningParser):
    """
    Reasoning parser for Kimi K2/K2.5 models.

    Kimi uses <think>...</think> for reasoning, but unlike DeepSeek R1,
    it may skip the </think> token and implicitly end reasoning by emitting
    <|tool_calls_section_begin|>. The base DeepSeekR1ReasoningParser does
    not handle this case, causing the tool parser to never be invoked and
    tool call markers to leak into the reasoning field as garbled output.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction."
            )

        # Check if thinking is disabled via chat_template_kwargs
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        # Default to thinking enabled if not specified
        thinking_enabled = True
        if thinking is not None:
            thinking_enabled = bool(thinking)
        if enable_thinking is not None:
            thinking_enabled = bool(enable_thinking)

        if not thinking_enabled:
            self._identity_parser = IdentityReasoningParser(
                tokenizer, *args, **kwargs
            )
        else:
            self._identity_parser = None

        # Token definitions
        self._start_token = "<think>"
        self._end_token = "</think>"
        self._tool_section_start_token = "<|tool_calls_section_begin|>"

        # Get token IDs
        self._start_token_id = self.vocab.get(self._start_token)
        self._end_token_id = self.vocab.get(self._end_token)
        self._tool_section_start_token_id = self.vocab.get(
            self._tool_section_start_token
        )

        if self._start_token_id is None or self._end_token_id is None:
            raise RuntimeError(
                "KimiK2ReasoningParser could not locate think start/end "
                "tokens in the tokenizer!"
            )

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end(input_ids)

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == self._start_token_id:
                return False
            if input_ids[i] == self._end_token_id:
                return True
            # Implicit reasoning end: tool section starts without </think>
            if (
                self._tool_section_start_token_id is not None
                and input_ids[i] == self._tool_section_start_token_id
            ):
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        if self._identity_parser is not None:
            return self._identity_parser.is_reasoning_end_streaming(
                input_ids, delta_ids
            )

        if self._end_token_id in delta_ids:
            return True
        return (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in delta_ids
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self._identity_parser is not None:
            return self._identity_parser.extract_content_ids(input_ids)

        # Check for explicit </think> end
        if self._end_token_id in input_ids:
            end_idx = (
                len(input_ids)
                - 1
                - input_ids[::-1].index(self._end_token_id)
            )
            return input_ids[end_idx + 1 :]

        # Check for implicit end via tool section start
        if (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in input_ids
        ):
            tool_idx = (
                len(input_ids)
                - 1
                - input_ids[::-1].index(self._tool_section_start_token_id)
            )
            # Include the tool section start token itself as content
            return input_ids[tool_idx:]

        return []

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        if self._identity_parser is not None:
            return self._identity_parser.extract_reasoning(
                model_output, request
            )

        # Consume <think> if present, otherwise start from beginning
        start_idx = model_output.find(self._start_token)
        if start_idx != -1:
            start_idx += len(self._start_token)
        else:
            start_idx = 0

        # Check for explicit </think>
        end_idx = model_output.find(self._end_token)
        if end_idx != -1:
            reasoning = model_output[start_idx:end_idx]
            content = model_output[end_idx + len(self._end_token) :]
            return reasoning, content or None

        # Check for implicit end via tool section
        tool_idx = model_output.find(self._tool_section_start_token)
        if tool_idx != -1:
            reasoning = model_output[start_idx:tool_idx]
            # Content includes the tool section marker for the tool parser
            content = model_output[tool_idx:]
            return reasoning, content or None

        # No end marker found - everything is reasoning
        return model_output[start_idx:], None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self._identity_parser is not None:
            return self._identity_parser.extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

        # Skip single special tokens
        if len(delta_token_ids) == 1 and delta_token_ids[0] in (
            self._start_token_id,
            self._end_token_id,
        ):
            return None

        # Check for explicit </think> in delta
        if self._end_token_id in delta_token_ids:
            end_index = delta_text.find(self._end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self._end_token) :]
            return DeltaMessage(
                reasoning=reasoning if reasoning else None,
                content=content if content else None,
            )

        # Check for implicit end via tool section start in delta
        if (
            self._tool_section_start_token_id is not None
            and self._tool_section_start_token_id in delta_token_ids
        ):
            tool_index = delta_text.find(self._tool_section_start_token)
            reasoning = delta_text[:tool_index]
            # Pass tool section marker onward as content for tool parser
            content = delta_text[tool_index:]
            return DeltaMessage(
                reasoning=reasoning if reasoning else None,
                content=content if content else None,
            )

        # Check context to determine if we're in reasoning or content
        if self._start_token_id in previous_token_ids:
            if self._end_token_id in previous_token_ids:
                # Past reasoning
                return DeltaMessage(content=delta_text)
            # Check for implicit end via tool section in previous
            if (
                self._tool_section_start_token_id is not None
                and self._tool_section_start_token_id in previous_token_ids
            ):
                return DeltaMessage(content=delta_text)
            # Still in reasoning
            return DeltaMessage(reasoning=delta_text)

        if self._start_token_id in delta_token_ids:
            # Start token in delta - beginning of reasoning
            start_index = delta_text.find(self._start_token)
            reasoning = delta_text[start_index + len(self._start_token) :]
            return DeltaMessage(
                reasoning=reasoning if reasoning else None
            )

        # No start token seen - default to reasoning
        return DeltaMessage(reasoning=delta_text)
