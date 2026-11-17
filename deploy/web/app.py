"""Hugging Face Space entry point for the TrialGuide demo (free CPU, no GPU needed).

One Space, two pages:
  /       the recorded demo (index.html, app.js, styles.css, data.json in this folder). It replays
          real Qwen3-8B decisions on 3 development patients, as described in the report.
  /live/  bring your own patient (trialguide.demo.live_app). On the free CPU tier the two models
          are replaced by a word-matching stand-in, so it shows the flow, not the models' decisions.
The TrialGuide code is installed from the GitHub repo (requirements.txt). Owner: Prakash.
"""

import urllib.request
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"

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

# a ZeroGPU Space refuses to start without one @spaces.GPU function; this one is never called
try:
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return None
except ImportError:          # running outside Hugging Face
    pass

app = FastAPI()

# the live page; fake=True is the word-matching stand-in (no GPU on the free tier)
app = gr.mount_gradio_app(app, build(RAW, fake=True, home="../").queue(), path="/live")

# the recorded demo: only these four files are served
STATIC = {"app.js": "text/javascript", "styles.css": "text/css", "data.json": "application/json"}


@app.get("/")
def home():
    return FileResponse(HERE / "index.html", media_type="text/html")


@app.get("/{name}")
def static_file(name: str):
    if name not in STATIC:
        return FileResponse(HERE / "index.html", media_type="text/html")
    return FileResponse(HERE / name, media_type=STATIC[name])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
