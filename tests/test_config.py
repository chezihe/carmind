from carmind.config import deep_merge, load_config


def test_deep_merge_overrides_nested_keys_only():
    base = {"runtime": {"device": "auto", "seed": 42}}
    override = {"runtime": {"device": "cuda"}}
    assert deep_merge(base, override) == {"runtime": {"device": "cuda", "seed": 42}}


def test_env_configs_load():
    local = load_config("local")
    runpod = load_config("runpod")
    assert local.data.max_samples == 100
    assert runpod.runtime.device == "cuda"
    assert runpod.runtime.seed == 42  # base 값이 유지됨
