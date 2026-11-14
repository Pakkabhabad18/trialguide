"""Hugging Face Space entry point for the TrialGuide demo (free CPU, no GPU needed).

The demo replays recorded Qwen3-8B checks (demo/demo_bundle.json), so it runs on the free tier.
The code comes from the GitHub repo (requirements.txt); the bundle and the answerability prior it was
recorded with are downloaded from the same repo on first start, so the Space holds no copy of them.
"""

import urllib.request
from pathlib import Path

REPO = "https://raw.githubusercontent.com/Pakkabhabad18/trialguide/main/"
# the bundle, and the prior the bundle's meta points to (the question choice depends on it)
FILES = ["demo/demo_bundle.json", "results/answerability_prior.json"]

for name in FILES:
    path = Path(name)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(REPO + name, path)

from trialguide.demo.app import build  # noqa: E402  (after the files exist)

# Free Hugging Face accounts get ZeroGPU hardware for Gradio Spaces, and a ZeroGPU Space refuses to
# start unless the app declares at least one @spaces.GPU function. The demo replays recorded outputs
# and needs no GPU, so this placeholder is declared but never called (it uses no GPU quota).
try:
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return None
except ImportError:          # running outside Hugging Face: nothing to declare
    pass

build(Path("demo/demo_bundle.json")).launch()
