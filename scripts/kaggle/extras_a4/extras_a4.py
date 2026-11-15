"""Kaggle notebook (post hoc): ablation A4, the rule checker without cited evidence.

Same as Ramana's risk-checks notebook (results/resplits/), which re-ran the frozen checker on the
expert-labelled rules of the 26 calibration and 26 test patients, but with --no-evidence: the checker's
prompt no longer asks it to cite sentences (checker/rule_checker.py, SYSTEM_PROMPT_NO_EVIDENCE), and an
uncited conclusion is not turned into "not enough information". Everything else is the same: Qwen3-8B 8-bit,
greedy, tools on, rule-targeted mask hiding 40%. Comparing it with results/resplits/ (same machine type)
shows what grounding in cited sentences does to harmful inferences. No doctor, so no HF_TOKEN is needed.
Calibration and test run in parallel, one T4 each; outputs go to /kaggle/working.
"""
import json, os, subprocess, sys, time, shutil
from pathlib import Path

BRANCH = "prakash"
REPO = "https://github.com/Pakkabhabad18/trialguide.git"
WORK = Path("/kaggle/working")
SRC = Path("/kaggle/trialguide")
LAMBDA_HAT = "0.7660011405599875"          # from results/dev_run/calibration.json (frozen)


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=kw.pop("check", True), **kw)


t0 = time.time()
sh(f"git clone -q --branch {BRANCH} {REPO} {SRC} && cd {SRC} && git log -1 --format='%H %s'")
COMMIT = subprocess.run(f"cd {SRC} && git rev-parse --short HEAD", shell=True, capture_output=True, text=True).stdout.strip()
os.chdir(SRC)
sh("pip install -q -U bitsandbytes accelerate nltk==3.9.2 rank-bm25==0.2.2 pyarrow && pip install -q -e . --no-deps")
sh("python -c \"import nltk; nltk.download('punkt_tab', quiet=True)\"")
sh("python -m trialguide.data.download")
os.environ["HF_HOME"] = "/tmp/hf"
sh("python -c \"from huggingface_hub import snapshot_download as s; "
   "print(s('Qwen/Qwen3-8B', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*']))\"")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; pip freeze > " + str(WORK / "pip_freeze.txt"))

import torch
n_gpu = torch.cuda.device_count()
print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)], flush=True)

base = {**os.environ, "LLM_HF_8BIT": "1", "LLM_BACKEND": "hf", "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(SRC / "src")}
jobs = {
    "cal": f"python -m trialguide.baselines.run_risk --split cal --fraction 0.4 --no-evidence --out {WORK}/cal",
    "test": (f"python -m trialguide.baselines.run_risk --split test --allow-test --fraction 0.4 --no-evidence "
             f"--thresholds B2=0 B5=0.9 B7={LAMBDA_HAT} --out {WORK}/test"),
}
procs = {}
for i, (name, cmd) in enumerate(jobs.items()):
    env = {**base, "LLM_CACHE": str(WORK / f"cache_{name}.jsonl"), "LLM_LOG": str(WORK / f"calls_{name}.jsonl"),
           "CUDA_VISIBLE_DEVICES": str(i if n_gpu > 1 else 0)}
    log = open(WORK / f"{name}.log", "w")
    if n_gpu > 1 or i == 0:
        procs[name] = (subprocess.Popen(cmd, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT), env, cmd)
if n_gpu < 2:                                   # one GPU: run the second job after the first
    first = procs["cal"][0]
    first.wait()
    env = {**base, "LLM_CACHE": str(WORK / "cache_test.jsonl"), "LLM_LOG": str(WORK / "calls_test.jsonl"),
           "CUDA_VISIBLE_DEVICES": "0"}
    procs["test"] = (subprocess.Popen(jobs["test"], shell=True, env=env,
                                      stdout=open(WORK / "test.log", "w"), stderr=subprocess.STDOUT), env, jobs["test"])

while any(p.poll() is None for p, _, _ in procs.values()):
    time.sleep(60)
    for name in procs:
        cache = WORK / f"cache_{name}.jsonl"
        n = sum(1 for _ in open(cache)) if cache.exists() else 0
        print(f"[{(time.time() - t0) / 60:5.1f} min] {name}: {n} model calls cached", flush=True)
rc = {n: p.returncode for n, (p, _, _) in procs.items()}
print("exit codes:", rc, "elapsed min:", round((time.time() - t0) / 60, 1), flush=True)
(WORK / "run_info.json").write_text(json.dumps({"commit": COMMIT, "exit_codes": rc, "gpus": n_gpu,
                                                 "elapsed_min": round((time.time() - t0) / 60, 1)}))
sys.exit(0 if all(v == 0 for v in rc.values()) else 1)
