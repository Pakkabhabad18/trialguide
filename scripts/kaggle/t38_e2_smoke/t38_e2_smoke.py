"""Kaggle kernel, T38 (exploratory): the frozen system on 10 Indian-affiliated case reports (E2 calibration half).

Runs only the Qwen3-8B checker (no doctor, no secret needed): the full and the 40% rule-masked note against the top 2 E2
trials per patient (about 710 checker calls), then B6/B2/B5 and rule recovery against the full-note checker.
Frozen code commit 55be3fe, plus ONLY src/trialguide/eval/e2_smoke.py and the trial snapshot taken from the branch below.
The two T4s each take every other sampled patient; the E2 test half is never read.
"""
import json, os, subprocess, sys, time
from pathlib import Path

COMMIT, BRANCH = "55be3fe", "murthy-T38-e2-smoke"
REPO = "https://github.com/Pakkabhabad18/trialguide.git"
WORK, SRC, TMP = Path("/kaggle/working"), Path("/kaggle/trialguide"), Path("/tmp/e2")
TMP.mkdir(exist_ok=True)
N, K, FRACTION = 10, 2, 0.4


def sh(cmd, check=True, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check, **kw)


t0 = time.time()
sh(f"git clone -q {REPO} {SRC} && cd {SRC} && git checkout -q {COMMIT} && git fetch -q origin {BRANCH} && "
   f"git checkout FETCH_HEAD -- src/trialguide/eval/e2_smoke.py results/e2_smoke/e2_trials_corpus.json && git log -1 --format='%H %s' {COMMIT}")
os.chdir(SRC)
sh("pip install -q -U bitsandbytes accelerate nltk==3.9.2 rank-bm25==0.2.2 pyarrow && pip install -q -e . --no-deps")
sh("python -c \"import nltk; nltk.download('punkt_tab', quiet=True)\"")
sh("python -m trialguide.data.download --with-pmc-patients")
os.environ["HF_HOME"] = "/tmp/hf"
sh("python -c \"from huggingface_hub import snapshot_download as s; "
   "print(s('Qwen/Qwen3-8B', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*']))\"")
sh("pip freeze > " + str(WORK / "pip_freeze.txt"))

import torch
n_gpu = torch.cuda.device_count()
print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)], flush=True)

procs = {}
for i, part in enumerate(("a", "b")):
    env = {**os.environ, "LLM_BACKEND": "hf", "LLM_HF_8BIT": "1", "LLM_HF_BATCH": "4", "PYTHONUNBUFFERED": "1",
           "PYTHONPATH": str(SRC / "src"), "CUDA_VISIBLE_DEVICES": str(i if n_gpu > 1 else 0),
           "LLM_CACHE": str(WORK / f"cache_{part}.jsonl"), "LLM_LOG": str(TMP / f"calls_{part}.jsonl")}
    cmd = (f"python -m trialguide.eval.e2_smoke --out {WORK}/e2_{part} --part {part} --n {N} --trials {K} "
           f"--fraction {FRACTION}")
    procs[part] = subprocess.Popen(cmd, shell=True, env=env, stdout=open(TMP / f"run_{part}.log", "w"), stderr=subprocess.STDOUT)
    if n_gpu < 2:
        procs[part].wait()

while any(p.poll() is None for p in procs.values()):
    time.sleep(120)
    for part in procs:
        c = WORK / f"cache_{part}.jsonl"
        print(f"[{(time.time() - t0) / 60:6.1f} min] part {part}: {sum(1 for _ in open(c)) if c.exists() else 0} model calls cached", flush=True)
rc = {p: pr.returncode for p, pr in procs.items()}
for part in procs:
    lines = (TMP / f"run_{part}.log").read_text(errors="replace").splitlines()
    (WORK / f"run_{part}_tail.log").write_text("\n".join(lines[-120:]))
(WORK / "run_info.json").write_text(json.dumps({"commit": COMMIT, "branch": BRANCH, "patients": N, "trials_per_patient": K,
                                                 "fraction": FRACTION, "exit_codes": rc, "gpus": n_gpu,
                                                 "elapsed_min": round((time.time() - t0) / 60, 1)}))
print("exit codes:", rc, "elapsed min:", round((time.time() - t0) / 60, 1), flush=True)
sys.exit(0 if all(v == 0 for v in rc.values()) else 1)
