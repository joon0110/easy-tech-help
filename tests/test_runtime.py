"""Prevent inherited chat penalties from altering extraction token scores."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from easy_tech_help.runtime import LocalRuntime, generation_settings


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


def test_rag_temporarily_disables_real_peft_adapter_and_restores_after_error(
    monkeypatch,
):
    peft = pytest.importorskip("peft")
    base = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    runtime = LocalRuntime.from_loaded(model, None, "cpu")
    layer = model.base_model.model.model.layers[0].self_attn.q_proj
    assert not layer.disable_adapters

    def fail(*args, **kwargs):
        assert layer.disable_adapters
        assert kwargs["repetition_penalty"] == 1.1
        raise RuntimeError("generation failed")

    monkeypatch.setattr(runtime, "_generate", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        runtime.generate_grounded([])
    assert not layer.disable_adapters
    monkeypatch.setattr(runtime, "_generate", lambda *a: layer.disable_adapters)
    assert runtime.generate([]) is False
