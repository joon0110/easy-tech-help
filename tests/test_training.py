"""Check actual PyTorch causal loss alignment without downloaded model weights."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from easy_tech_help.training import completion_loss, decision_spans, selection_key


def test_completion_loss_matches_masked_full_causal_loss():
    torch.manual_seed(42)
    model = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            attention_dropout=0.0,
        )
    )
    model.eval()
    example = {"ids": [1, 5, 4, 8, 2, 9], "start": 3}
    ids = torch.tensor([example["ids"]])
    labels = ids.clone()
    labels[:, :3] = -100
    expected = model(input_ids=ids, labels=labels, use_cache=False).loss
    actual, count = completion_loss(model, example, "cpu")
    assert count == 3
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(model.lm_head.weight.grad).all()
    assert model.lm_head.weight.grad.abs().sum() > 0


def test_weighted_completion_loss_matches_manual_token_objective():
    model = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    ).eval()
    example = {"ids": [1, 5, 4, 8, 2, 9], "start": 3, "weights": [4.0, 1.0, 4.0]}
    ids = torch.tensor([example["ids"]])
    logits = model(input_ids=ids, use_cache=False).logits[0, 2:5].float()
    losses = torch.nn.functional.cross_entropy(logits, ids[0, 3:], reduction="none")
    expected = (losses * torch.tensor(example["weights"])).sum() / 9
    actual, total_weight = completion_loss(model, example, "cpu")
    assert total_weight == 9
    torch.testing.assert_close(actual, expected)


def test_decision_spans_do_not_confuse_brackets_inside_quoted_evidence():
    import json

    content = json.dumps(
        {
            "category": "message",
            "signals": [
                {"signal": "support_phone_number", "evidence": "Call [PHONE]"},
                {"signal": "visible_link", "evidence": "https://help.example"},
            ],
            "issues": [],
        }
    )
    spans = decision_spans(content)
    for token in ["message", "support_phone_number", "visible_link"]:
        position = content.index(token)
        assert any(start <= position < end for start, end in spans)
    for token in ["[PHONE]", "https://help.example"]:
        position = content.index(token)
        assert not any(start <= position < end for start, end in spans)
    position = content.index('], "issues"')
    assert any(start <= position < end for start, end in spans)


def test_checkpoint_selection_prefers_answers_over_lower_teacher_forced_loss():
    fewer = {
        "observation_matches": 10,
        "ordinary_with_extra_signals": 0,
        "signal_f1": 0.9,
    }
    more = fewer | {"observation_matches": 20}
    assert selection_key(more, 0.2) > selection_key(fewer, 0.01)
    assert selection_key(more, 0.2) > selection_key(
        more | {"ordinary_with_extra_signals": 1}, 0.01
    )
