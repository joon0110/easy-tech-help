"""Prevent inherited chat penalties from altering extraction token scores."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from easy_tech_help.runtime import generation_settings


def test_extraction_uses_raw_greedy_scores_despite_model_chat_defaults():
    torch.manual_seed(7)
    model = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).eval()
    # These are the nonneutral defaults in the pinned Qwen model download.
    model.generation_config.repetition_penalty = 1.1
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.8
    model.generation_config.top_k = 20
    config = transformers.GenerationConfig(
        **generation_settings(1),
        eos_token_id=2,
        pad_token_id=0,
        return_dict_in_generate=True,
        output_logits=True,
        output_scores=True,
    )
    ids = torch.tensor([[1, 4, 7, 4, 8]])
    with torch.inference_mode():
        result = model.generate(input_ids=ids, generation_config=config)
    # A repetition processor would change logits for already-seen input tokens.
    torch.testing.assert_close(result.scores[0], result.logits[0], rtol=0, atol=0)
    assert result.sequences[0, -1] == result.logits[0][0].argmax()
