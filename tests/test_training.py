"""Check actual PyTorch causal loss alignment without downloaded model weights."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from easy_tech_help.training import completion_loss


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
