"""Kaggle kernel, T35 phase 1: the 20% masking run on a seeded subset of the test patients, without the doctor.

What it does: builds the B6 manifest for 24 of the 73 test patients (seed 20261030, proportional to cohort,
4 trials per expert label, exactly as the 40% test run) at 20% rule-targeted masking, then runs the checker once
over every rule of every candidate trial and applies the policies that never ask (B6, B2, B5). The two T4s each
take half of the patients. The LLM cache (every checker call) is kept in /kaggle/working so phase 2 (B7, B4 with
the Llama doctor) re-uses it. Frozen code commit 55be3fe, Qwen3-8B 8-bit, greedy, same as results/test_run.
"""
import json, os, subprocess, sys, time
from pathlib import Path

COMMIT, N_PATIENTS, FRACTION = "55be3fe", 24, 0.2
REPO = "https://github.com/Pakkabhabad18/trialguide.git"
WORK, SRC, TMP = Path("/kaggle/working"), Path("/kaggle/trialguide"), Path("/tmp/t35")
TMP.mkdir(exist_ok=True)


def sh(cmd, check=True, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, **kw)


t0 = time.time()
sh(f"git clone -q {REPO} {SRC} && cd {SRC} && git checkout -q {COMMIT} && git log -1 --format='%H %s'")
os.chdir(SRC)
sh("pip install -q -U bitsandbytes accelerate nltk==3.9.2 rank-bm25==0.2.2 pyarrow && pip install -q -e . --no-deps")
sh("python -c \"import nltk; nltk.download('punkt_tab', quiet=True)\"")
sh("python -m trialguide.data.download")
os.environ["HF_HOME"] = "/tmp/hf"
sh("python -c \"from huggingface_hub import snapshot_download as s; "
   "print(s('Qwen/Qwen3-8B', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*']))\"")
sh("pip freeze > " + str(WORK / "pip_freeze.txt"))

import torch
n_gpu = torch.cuda.device_count()
print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)], flush=True)

# the manifest: the same code and seed as the 40% test run, 24 patients, 60% masking
sh(f"python -m trialguide.baselines.b6 --raw data/raw --out {WORK}/data --n {N_PATIENTS} --split test "
   f"--allow-test --per-label 4 --fraction {FRACTION} > {TMP}/manifest.log 2>&1")
manifest = json.load(open(WORK / "data" / "manifest.json"))
ids = sorted(manifest["patients"])
halves = {"a": ids[0::2], "b": ids[1::2]}
for h, ps in halves.items():
    json.dump({**manifest, "patients": {p: manifest["patients"][p] for p in ps}}, open(WORK / f"manifest_{h}.json", "w"), indent=1)
print({h: len(ps) for h, ps in halves.items()}, flush=True)

procs = {}
for i, h in enumerate(halves):
    env = {**os.environ, "LLM_BACKEND": "hf", "LLM_HF_8BIT": "1", "LLM_HF_BATCH": "4", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": str(SRC / "src"), "CUDA_VISIBLE_DEVICES": str(i if n_gpu > 1 else 0),
           "LLM_CACHE": str(WORK / f"cache_{h}.jsonl"), "LLM_LOG": str(TMP / f"calls_{h}.jsonl")}
    cmd = (f"python -m trialguide.baselines.run_systems --manifest {WORK}/manifest_{h}.json --allow-test "
           f"--conditions rule{round(FRACTION * 100)} --out {WORK}/systems_{h}")
    procs[h] = subprocess.Popen(cmd, shell=True, env=env, stdout=open(TMP / f"run_{h}.log", "w"), stderr=subprocess.STDOUT)
    if n_gpu < 2:
        procs[h].wait()

while any(p.poll() is None for p in procs.values()):
    time.sleep(120)
    for h in procs:
        c = WORK / f"cache_{h}.jsonl"
        print(f"[{(time.time() - t0) / 60:6.1f} min] half {h}: {sum(1 for _ in open(c)) if c.exists() else 0} model calls cached", flush=True)
rc = {h: p.returncode for h, p in procs.items()}
for h in procs:                                     # keep only the tail of each (noisy) log
    lines = (TMP / f"run_{h}.log").read_text(errors="replace").splitlines()
    (WORK / f"run_{h}_tail.log").write_text("\n".join(lines[-150:]))
(WORK / "run_info.json").write_text(json.dumps({"commit": COMMIT, "patients": N_PATIENTS, "fraction": FRACTION,
                                                 "exit_codes": rc, "gpus": n_gpu,
                                                 "elapsed_min": round((time.time() - t0) / 60, 1)}))
print("exit codes:", rc, "elapsed min:", round((time.time() - t0) / 60, 1), flush=True)
sys.exit(0 if all(v == 0 for v in rc.values()) else 1)
