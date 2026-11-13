import json

from trialguide.llm import client as llm


def _isolated_client(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_PATH", str(tmp_path / "cache.jsonl"))
    monkeypatch.setattr(llm, "LOG_PATH", str(tmp_path / "calls.jsonl"))
    monkeypatch.setattr(llm, "BACKEND", "openai")
    monkeypatch.setattr(llm, "_cache", None)


def test_cache_hit_and_miss_and_backward_compatible_call(tmp_path, monkeypatch):
    _isolated_client(tmp_path, monkeypatch)
    calls = []

    def complete(messages, model, temperature, logprobs=False, seed=None, top_p=1.0):
        calls.append((messages, model, temperature, logprobs, seed, top_p))
        return "reply", {"served_model": model, "system_fingerprint": "fp", "model_revision": "rev"}

    monkeypatch.setattr(llm, "_complete_api", complete)
    prompt = [{"role": "user", "content": "visible text"}]
    assert llm.chat(prompt, "Qwen/Qwen3-8B") == "reply"  # original caller shape
    assert llm.chat(prompt, "Qwen/Qwen3-8B") == "reply"
    assert len(calls) == 1
    assert len((tmp_path / "cache.jsonl").read_text().splitlines()) == 1


def test_cache_separates_prompt_temperature_seed_and_top_p(tmp_path, monkeypatch):
    _isolated_client(tmp_path, monkeypatch)
    calls = []

    def complete(*args, **kwargs):
        calls.append((args, kwargs))
        return str(len(calls)), {"served_model": "Qwen/Qwen3-8B"}

    monkeypatch.setattr(llm, "_complete_api", complete)
    base = [{"role": "user", "content": "prompt one"}]
    assert llm.chat(base, "Qwen/Qwen3-8B", temperature=0.7, seed=10, top_p=0.9) == "1"
    assert llm.chat(base, "Qwen/Qwen3-8B", temperature=0.7, seed=10, top_p=0.9) == "1"
    assert llm.chat([{**base[0], "content": "prompt two"}], "Qwen/Qwen3-8B",
                    temperature=0.7, seed=10, top_p=0.9) == "2"
    assert llm.chat(base, "Qwen/Qwen3-8B", temperature=0.8, seed=10, top_p=0.9) == "3"
    assert llm.chat(base, "Qwen/Qwen3-8B", temperature=0.7, seed=11, top_p=0.9) == "4"
    assert llm.chat(base, "Qwen/Qwen3-8B", temperature=0.7, seed=10, top_p=0.8) == "5"
    assert len(calls) == 5


def test_generation_log_records_explicit_reproducibility_metadata(tmp_path, monkeypatch):
    _isolated_client(tmp_path, monkeypatch)
    monkeypatch.setattr(llm, "_complete_api", lambda *a, **k: (
        "answer", {"served_model": "Qwen/Qwen3-8B", "system_fingerprint": "fingerprint"}))
    llm.chat([{"role": "user", "content": "visible"}], "Qwen/Qwen3-8B",
             temperature=0.6, seed=42, top_p=0.85)
    log = json.loads((tmp_path / "calls.jsonl").read_text())
    assert log["model"] == "Qwen/Qwen3-8B"
    assert log["served_model"] == "Qwen/Qwen3-8B"
    assert log["seed"] == 42 and log["temperature"] == 0.6 and log["top_p"] == 0.85
    assert log["system_fingerprint"] == "fingerprint" and log["date"]


def test_invalid_sampling_configuration_rejected(tmp_path, monkeypatch):
    _isolated_client(tmp_path, monkeypatch)
    for options in ({"temperature": -0.1}, {"top_p": 0}, {"top_p": 1.1}):
        try:
            llm.chat([], "model", **options)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid sampling config: {options}")
