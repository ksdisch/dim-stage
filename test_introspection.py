"""test_introspection.py — analytic checks of the protocol-2 conventions (no downloads).

The steering-scale computation, the question-turn locator, the drift guard,
and the MRR metric are each verified on rigged inputs before any real run.
"""
from __future__ import annotations

import json

import pytest
import torch

import m1_introspection as intro
from test_readability import RiggedSubject


def test_load_introspection_guards_against_drift(tmp_path):
    bad = tmp_path / "verbal-introspection.json"
    bad.write_text(
        json.dumps(
            {
                "intro_prompt": [{"role": "user", "content": "x"}],
                "prefills": {"default": "a", "word": "b"},
                "concepts": [{"name": "n", "surface": "s"}],
            }
        )
    )
    with pytest.raises(ValueError, match="drifted"):
        intro.load_introspection(str(bad))
    with pytest.raises(FileNotFoundError, match="refetch"):
        intro.load_introspection(str(tmp_path / "missing.json"))


def test_median_reciprocal_rank():
    assert intro.median_reciprocal_rank([1, 2, 10]) == 0.5
    assert intro.median_reciprocal_rank([1, 4]) == 0.625  # (1 + 0.25) / 2
    assert intro.median_reciprocal_rank([1]) == 1.0


def test_mean_residual_norms_exact_on_rigged_subject():
    # Layer 0 residual = 3e0 everywhere, layer 1 = 5e1 everywhere: the mean L2
    # norm over any valid positions is exactly 3 and 5. A prompt shorter than
    # skip_first+2 contributes nothing (the fitter's skip rule).
    e = torch.eye(4)
    subject = RiggedSubject(
        [3.0 * e[0].repeat(24, 1), 5.0 * e[1].repeat(24, 1)], torch.zeros(11, 4)
    )
    norms = intro.mean_residual_norms(subject, ["x" * 20, "y" * 5], [0, 1])
    assert norms == {0: 3.0, 1: 5.0}
    with pytest.raises(ValueError, match="long enough"):
        intro.mean_residual_norms(subject, ["y" * 5], [0])


class ConcatChatTokenizer:
    """A minimal chat template that concatenates turns: each message becomes
    [1, *content codes, 2]; the generation prompt is a trailing [3]."""

    chat_template = "stub"

    def apply_chat_template(self, msgs, add_generation_prompt, return_dict):
        ids: list[int] = []
        for m in msgs:
            ids += [1] + [ord(c) % 50 + 10 for c in m["content"]] + [2]
        return ids + ([3] if add_generation_prompt else [])

    def __call__(self, text, add_special_tokens=False):
        class Enc:
            input_ids = [ord(c) % 50 + 10 for c in text]

        return Enc()


class ReorderingChatTokenizer(ConcatChatTokenizer):
    """A pathological template that rewrites earlier turns — the prefix
    property fails and build_prompt must exit INVALID."""

    def apply_chat_template(self, msgs, add_generation_prompt, return_dict):
        return super().apply_chat_template(list(reversed(msgs)), add_generation_prompt, return_dict)


class FakeSubject:
    _input_device = "cpu"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


MESSAGES = [
    {"role": "user", "content": "abc"},
    {"role": "assistant", "content": "d"},
    {"role": "user", "content": "ef"},
]


def test_build_prompt_locates_the_question_turn():
    subject = FakeSubject(ConcatChatTokenizer())
    input_ids, positions = intro.build_prompt(subject, MESSAGES, "gh")
    # Blocks: [1,a,b,c,2] [1,d,2] [1,e,f,2] + [3] + prefill [g,h]
    two_len, three_len = 5 + 3, 5 + 3 + 4
    assert positions == list(range(two_len, three_len))  # the third block
    assert input_ids.shape == (1, three_len + 1 + 2)
    tok = ConcatChatTokenizer()
    assert input_ids[0, -2:].tolist() == tok("gh").input_ids  # prefill verbatim


def test_build_prompt_rejects_a_non_concatenating_template():
    subject = FakeSubject(ReorderingChatTokenizer())
    with pytest.raises(SystemExit) as exc:
        intro.build_prompt(subject, MESSAGES, "gh")
    assert exc.value.code == 2
