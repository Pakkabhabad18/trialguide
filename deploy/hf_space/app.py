"""Hugging Face Space entry point for TrialGuide live: try your own patient (no GPU needed).

Paste or upload a note, or start from a few facts, and answer TrialGuide's questions as the doctor.
The Space has no GPU for the two 8B models. While the Kaggle notebook trialguide_live_kaggle.ipynb runs,
the page uses the real models there; otherwise it runs the word-matching stand-in, and the badge says which. The recorded demo with real Qwen3-8B decisions is the static page at
huggingface.co/spaces/Pakkabhabad18/trialguide-demo. The code is installed from the GitHub repo
(requirements.txt). Owner: Prakash.
"""

import urllib.request
from pathlib import Path

RAW = Path("data/raw")

# the two trial collections the live page searches, from the pinned TrialGPT sources
TRIALGPT = "https://raw.githubusercontent.com/ncbi-nlp/TrialGPT/1b3242cd50c153a9c324661d4a0bd94fc68a4226/dataset"
CORPORA = {
    "sigir": f"{TRIALGPT}/sigir/corpus.jsonl",
    "trec_2021": "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_corpus.jsonl",
}
for cohort, url in CORPORA.items():
    path = RAW / "trialgpt" / cohort / "corpus.jsonl"
    if not path.exists():                      # downloaded once, on the first start
        path.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", cohort, flush=True)
        urllib.request.urlretrieve(url, path)

from trialguide.demo.live_app import build  # noqa: E402  (after the files exist)

# Free Hugging Face accounts get ZeroGPU hardware for Gradio Spaces, and a ZeroGPU Space refuses to
# start unless the app declares at least one @spaces.GPU function. This placeholder is never called.
try:
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return None
except ImportError:          # running outside Hugging Face
    pass

HOME = "https://huggingface.co/spaces/Pakkabhabad18/trialguide-demo"
# remote=True: while the Kaggle GPU notebook is running (it checks in every 2 minutes), rule checks
# go to the real Qwen3-8B and Llama-3.1-8B there; otherwise the word-matching stand-in is used.
build(RAW, fake=True, home=HOME, remote=True).queue().launch()
