"""Download the raw datasets (T3). Standard library only, safe to re-run.

    python -m trialguide.data.download                 # core files, ~0.4 GB
    python -m trialguide.data.download --with-trial-info   # + trial_info.json, 1.2 GB
    python -m trialguide.data.download --update-checksums  # maintainers only

Files land in <root>/raw/ (default root: ./data, which is git-ignored). Each file is
checked against the SHA-256 in checksums.json (committed), so everyone on the team
works on byte-identical data. TrialGPT is pinned to one commit and the annotation
dataset to one Hugging Face revision, so the files cannot change under us.
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

# Pinned upstream versions, so every teammate gets byte-identical files
TRIALGPT_COMMIT = "1b3242cd50c153a9c324661d4a0bd94fc68a4226"
ANNOTATIONS_REVISION = "1cfcafde94a1560a33b4addc1638664fe28fc059"

_GH = f"https://raw.githubusercontent.com/ncbi-nlp/TrialGPT/{TRIALGPT_COMMIT}/dataset"
_NCBI = "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT"
_HF = f"https://huggingface.co/datasets/ncbi/TrialGPT-Criterion-Annotations/resolve/{ANNOTATIONS_REVISION}"

CHECKSUM_FILE = Path(__file__).with_name("checksums.json")


def manifest(with_trial_info: bool = False) -> dict:
    """Map of destination path (relative to raw/) -> source URL."""
    files = {}
    # Patients (queries), expert trial labels (qrels), cached keywords, candidate trials
    for corpus in ("sigir", "trec_2021", "trec_2022"):
        for name in ("queries.jsonl", "id2queries.json", "retrieved_trials.json", "qrels/test.tsv"):
            files[f"trialgpt/{corpus}/{name}"] = f"{_GH}/{corpus}/{name}"
    # Trial texts: SIGIR ships in the repo, the two TREC corpora come from NCBI
    files["trialgpt/sigir/corpus.jsonl"] = f"{_GH}/sigir/corpus.jsonl"
    files["trialgpt/trec_2021/corpus.jsonl"] = f"{_NCBI}/trec_2021_corpus.jsonl"
    files["trialgpt/trec_2022/corpus.jsonl"] = f"{_NCBI}/trec_2022_corpus.jsonl"
    # The 1,020 expert criterion-level labels
    files["annotations/criterion_annotations.parquet"] = f"{_HF}/data/train-00000-of-00001.parquet"
    if with_trial_info:
        files["trialgpt/trial_info.json"] = f"{_NCBI}/trial_info.json"
    return files


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch(url: str, dest: Path, retries: int = 3) -> None:
    """Download to a .part file and rename, so an interrupted run never leaves a half file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
                while block := r.read(1 << 20):
                    f.write(block)
            part.replace(dest)
            return
        except Exception as e:  # network errors vary; retry then give up loudly
            print(f"  attempt {attempt}/{retries} failed: {e}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def download(root: Path, with_trial_info: bool = False, update_checksums: bool = False) -> int:
    """Fetch every missing file and verify all of them. Returns the number of problems."""
    raw = root / "raw"
    known = json.loads(CHECKSUM_FILE.read_text()) if CHECKSUM_FILE.exists() else {}
    problems, computed = 0, {}
    for rel, url in manifest(with_trial_info).items():
        dest = raw / rel
        if not dest.exists():
            print(f"downloading {rel}")
            _fetch(url, dest)
        digest = sha256_of(dest)
        computed[rel] = digest
        if update_checksums or rel not in known:
            status = "recorded"
        elif known[rel] == digest:
            status = "ok"
        else:
            status, problems = "CHECKSUM MISMATCH", problems + 1
        print(f"  {status:17s} {rel}")
    if update_checksums:
        merged = {**known, **computed}
        CHECKSUM_FILE.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(f"wrote {CHECKSUM_FILE}")
    return problems


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default="data", help="data folder (default: ./data)")
    p.add_argument("--with-trial-info", action="store_true", help="also fetch trial_info.json (1.2 GB)")
    p.add_argument("--update-checksums", action="store_true", help="rewrite checksums.json from the files on disk")
    args = p.parse_args(argv)
    return 1 if download(Path(args.root), args.with_trial_info, args.update_checksums) else 0


if __name__ == "__main__":
    sys.exit(main())
