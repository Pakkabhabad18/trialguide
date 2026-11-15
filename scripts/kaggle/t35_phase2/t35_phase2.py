"""Kaggle kernel, T35 phase 2: B7 and B4 at 60% masking on the same 24 test patients, with the Llama doctor.

Needs phase 1 (kernel `trialguide-t35-phase1`, attached as a kernel source) for its manifests and its LLM cache,
and the Kaggle secret HF_TOKEN of an account that has been approved for the gated meta-llama/Llama-3.1-8B-Instruct.
It checks that access FIRST and exits (code 2 or 3) before downloading or using any GPU if it is missing.

Same frozen code (55be3fe), Qwen3-8B checker and Llama-3.1-8B-Instruct doctor, both 8-bit and greedy, lambda-hat
0.766, answerability prior from results/answerability_prior.json. Each half of the patients runs on one T4,
alternating the checker and the doctor in rounds exactly as scripts/run_asking.sh does. LLM_MAX_TOKENS is left at its
default (as in phase 1) so phase 1's cached checker calls are re-used; it must be the same in every round, because it is
part of the cache key.
"""
import json, os, shutil, subprocess, sys, threading, time, urllib.error, urllib.request
from pathlib import Path

COMMIT, LAMBDA_HAT, MAX_ROUNDS = "55be3fe", "0.7660011405599875", 14
# FALLBACK, OFF by default: load the doctor from the ungated NousResearch copy of Llama-3.1-8B-Instruct instead of the
# gated official repo. Its weight shards are byte-identical (SHA-256) to the official ones, but its chat template is
# different, so the doctor's prompt differs slightly from the main runs. Only switch this on by a deliberate decision.
DOCTOR_FROM_MIRROR = False
MIRROR = "NousResearch/Meta-Llama-3.1-8B-Instruct"
REPO = "https://github.com/Pakkabhabad18/trialguide.git"
WORK, SRC, TMP = Path("/kaggle/working"), Path("/kaggle/trialguide"), Path("/tmp/t35")
TMP.mkdir(exist_ok=True)
CHECKER, DOCTOR = "Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct"


def sh(cmd, check=True, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, **kw)


def alias_hf_repo(src, alias, hf_home, patterns=("*.json", "*.safetensors", "*.txt", "tokenizer*")):
    """Download `src` into the HF cache and expose its snapshot under the cache name of `alias`, so that
    from_pretrained(alias) finds it offline. Files are symlinks to the same blobs (no second copy)."""
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(src, cache_dir=f"{hf_home}/hub", allow_patterns=list(patterns)))
    sha = snap.name
    root = Path(hf_home) / "hub" / ("models--" + alias.replace("/", "--"))
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text(sha)
    dest = root / "snapshots" / sha
    for f in snap.rglob("*"):
        if f.is_file():
            target = dest / f.relative_to(snap)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    target.symlink_to(os.path.realpath(f))
                except OSError:                          # no symlink right (e.g. Windows): copy instead
                    shutil.copy2(os.path.realpath(f), target)
    return dest


# 0. fail fast: is the gated Llama repo readable with the attached secret? (the token is never printed)
def check_hf_access():
    """Exit before any download or GPU use if the gated official repo is not readable with the Kaggle secret."""
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print("NO_HF_TOKEN: attach the secret HF_TOKEN to this notebook (Add-ons > Secrets), then run it again.", type(e).__name__)
        sys.exit(2)
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://huggingface.co/{DOCTOR}/resolve/main/config.json", headers={"Authorization": f"Bearer {token}"}), timeout=30)
    except urllib.error.HTTPError as e:
        print(f"HF_ACCESS_DENIED: HTTP {e.code} for {DOCTOR}. The account behind HF_TOKEN is not approved for this model.")
        sys.exit(3)
    os.environ["HF_TOKEN"] = token                       # never printed
    print("Hugging Face access to", DOCTOR, "confirmed", flush=True)


if DOCTOR_FROM_MIRROR:
    print("DOCTOR_FROM_MIRROR is on: the doctor is the community copy", MIRROR, "(identical weights, different chat template)", flush=True)
else:
    check_hf_access()

t0 = time.time()
inp = next((p.parent for p in Path("/kaggle/input").rglob("cache_a.jsonl")), None)
if inp is None:
    print("PHASE1_OUTPUT_MISSING: attach kernel trialguide-t35-phase1 as a source"); sys.exit(4)
print("phase 1 output found at", inp, flush=True)
for f in ("manifest_a.json", "manifest_b.json", "cache_a.jsonl", "cache_b.jsonl"):
    sh(f"cp {inp}/{f} {WORK}/{f}")

sh(f"git clone -q {REPO} {SRC} && cd {SRC} && git checkout -q {COMMIT} && git log -1 --format='%H %s'")
os.chdir(SRC)
sh("pip install -q -U bitsandbytes accelerate nltk==3.9.2 rank-bm25==0.2.2 pyarrow && pip install -q -e . --no-deps")
sh("python -c \"import nltk; nltk.download('punkt_tab', quiet=True)\"")
sh("python -m trialguide.data.download")
os.environ["HF_HOME"] = "/tmp/hf"
sh(f"python -c \"from huggingface_hub import snapshot_download as s; print(s('{CHECKER}', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*']))\"")
if DOCTOR_FROM_MIRROR:
    print("doctor exposed from the mirror at", alias_hf_repo(MIRROR, DOCTOR, "/tmp/hf"), flush=True)
else:
    sh(f"python -c \"from huggingface_hub import snapshot_download as s; print(s('{DOCTOR}', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*']))\"")
sh("pip freeze > " + str(WORK / "pip_freeze.txt"))

import torch
n_gpu = torch.cuda.device_count()
print("GPUs:", n_gpu, flush=True)
results = {}


def run_half(h, gpu):
    env = {**os.environ, "LLM_BACKEND": "hf", "LLM_HF_8BIT": "1", "LLM_HF_BATCH": "8", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": str(SRC / "src"), "CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "LLM_CACHE": str(WORK / f"cache_{h}.jsonl"), "LLM_LOG": str(TMP / f"calls_{h}.jsonl")}
    out = WORK / f"asking_{h}"
    for rnd in range(1, MAX_ROUNDS + 1):
        model = CHECKER if rnd % 2 else DOCTOR
        cmd = (f"python -m trialguide.baselines.run_asking --manifest {WORK}/manifest_{h}.json --systems B7 B4 "
               f"--threshold {LAMBDA_HAT} --out {out} --allow-test")
        with open(TMP / f"run_{h}.log", "a") as log:
            log.write(f"\n=== round {rnd} {model}\n")
            rc = subprocess.run(cmd, shell=True, env={**env, "LLM_ONLY_MODEL": model}, stdout=log, stderr=subprocess.STDOUT).returncode
        status = out / "status.json"
        waiting = json.load(open(status))["waiting"] if status.exists() else None
        print(f"[{(time.time() - t0) / 60:6.1f} min] half {h} round {rnd} ({model}) rc={rc} "
              f"conversations waiting={None if waiting is None else len(waiting)}", flush=True)
        if rc != 0:
            results[h] = f"error in round {rnd}"; return
        if waiting is not None and not waiting and (out / "summary.json").exists():
            results[h] = "finished"; return
    results[h] = "stopped with conversations still waiting"


threads = [threading.Thread(target=run_half, args=(h, i if n_gpu > 1 else 0)) for i, h in enumerate(("a", "b"))]
for t in threads:
    t.start()
    if n_gpu < 2:
        t.join()
for t in threads:
    t.join()
for h in ("a", "b"):
    lines = (TMP / f"run_{h}.log").read_text(errors="replace").splitlines()
    (WORK / f"run_{h}_tail.log").write_text("\n".join(lines[-150:]))
(WORK / "run_info.json").write_text(json.dumps({"commit": COMMIT, "results": results, "gpus": n_gpu,
                                                 "elapsed_min": round((time.time() - t0) / 60, 1)}))
print(results, "elapsed min:", round((time.time() - t0) / 60, 1), flush=True)
sys.exit(0 if all(v == "finished" for v in results.values()) else 1)
