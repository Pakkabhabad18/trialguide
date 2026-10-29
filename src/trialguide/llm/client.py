"""
Shared LLM client (Team Zenith). Written for the TrialGPT reproduction (T4);
the team's shared client is Vidur's T5, so this may be merged into or replaced by it.

Backends (env LLM_BACKEND):
	hf     - default: in-process Hugging Face transformers, greedy decoding;
	         concurrent calls from worker threads are batched (LLM_HF_BATCH)
	openai - any OpenAI-compatible server, e.g. a local vLLM server
	         (OPENAI_BASE_URL, default http://localhost:8000/v1)
	azure  - original TrialGPT setup (OPENAI_ENDPOINT, OPENAI_API_KEY)

Every call is cached on disk (keyed by model + messages + params) and logged
with model, date and seed, so reruns are free and every result is traceable.
"""

import datetime
import hashlib
import json
import os
import queue
import re
import threading

BACKEND = os.getenv("LLM_BACKEND", "hf")
SEED = int(os.getenv("LLM_SEED", "0"))
# Qwen3 thinks by default; TrialGPT's prompts expect a direct JSON answer.
DISABLE_THINKING = os.getenv("LLM_DISABLE_THINKING", "1") == "1"
# the original code sets no limit; long exclusion lists need more than 4096
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))
CACHE_PATH = os.getenv("LLM_CACHE", "outputs/llm_cache.jsonl")
LOG_PATH = os.getenv("LLM_LOG", "outputs/llm_calls.jsonl")
HF_BATCH = int(os.getenv("LLM_HF_BATCH", "4"))

if BACKEND == "azure":
	from openai import AzureOpenAI
	client = AzureOpenAI(
		api_version="2023-09-01-preview",
		azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
		api_key=os.getenv("OPENAI_API_KEY"),
	)
elif BACKEND == "openai":
	from openai import OpenAI
	client = OpenAI(
		base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
		api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
		timeout=600,
	)

_lock = threading.Lock()
_cache = None


def _load_cache():
	global _cache
	if _cache is None:
		_cache = {}
		if os.path.exists(CACHE_PATH):
			with open(CACHE_PATH) as f:
				for line in f:
					entry = json.loads(line)
					_cache[entry["key"]] = entry["output"]
	return _cache


def _append(path, entry):
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with open(path, "a") as f:
		f.write(json.dumps(entry) + "\n")


def _complete_api(messages, model, temperature):
	kwargs = dict(model=model, messages=messages, temperature=temperature)
	if BACKEND == "openai":
		kwargs["seed"] = SEED
		kwargs["max_tokens"] = MAX_TOKENS
		if DISABLE_THINKING:
			kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

	response = client.chat.completions.create(**kwargs)
	return response.choices[0].message.content, {
		"served_model": response.model,
		"system_fingerprint": getattr(response, "system_fingerprint", None),
		"finish_reason": response.choices[0].finish_reason,
		"prompt_tokens": getattr(response.usage, "prompt_tokens", None),
		"completion_tokens": getattr(response.usage, "completion_tokens", None),
	}


class _HFBatcher:
	"""Loads one model and serves requests from many threads in small batches."""

	def __init__(self, model_name):
		import torch
		from transformers import AutoModelForCausalLM, AutoTokenizer

		self.torch = torch
		self.model_name = model_name
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
		if self.tokenizer.pad_token is None:
			# Llama tokenizers define no pad token; batching needs one
			self.tokenizer.pad_token = self.tokenizer.eos_token
		# cap the weights per GPU so they split evenly and every GPU keeps room
		# for the KV cache (otherwise the last GPU fills up and runs out of memory)
		max_memory = None
		if os.getenv("LLM_HF_MAX_MEMORY") and torch.cuda.device_count() > 1:
			max_memory = {i: os.getenv("LLM_HF_MAX_MEMORY") for i in range(torch.cuda.device_count())}
		self.model = AutoModelForCausalLM.from_pretrained(
			model_name, dtype=torch.bfloat16, device_map="auto", max_memory=max_memory
		).eval()
		print("hf device map:", sorted(set(self.model.hf_device_map.values()), key=str), flush=True)
		self.revision = getattr(self.model.config, "_commit_hash", None)
		self.requests = queue.Queue()
		threading.Thread(target=self._loop, daemon=True).start()

	def submit(self, messages):
		done = threading.Event()
		slot = {"messages": messages, "done": done}
		self.requests.put(slot)
		done.wait()
		if "error" in slot:
			raise RuntimeError(slot["error"])
		return slot["output"], slot["meta"]

	def _try_generate(self, batch):
		"""Returns None on success, else an error message.

		Only the message is kept: an exception object holds its traceback, whose
		frames hold the GPU tensors, so keeping it would leak that memory.
		"""
		try:
			self._generate(batch)
			return None
		except self.torch.cuda.OutOfMemoryError as e:
			error = f"CUDA out of memory: {str(e).splitlines()[0]}"
		except Exception as e:
			error = f"{type(e).__name__}: {e}"
		# free memory only after leaving the except block (the traceback is gone)
		self.torch.cuda.empty_cache()
		return error

	def _loop(self):
		while True:
			batch = [self.requests.get()]
			try:
				while len(batch) < HF_BATCH:
					batch.append(self.requests.get(timeout=0.5))
			except queue.Empty:
				pass
			error = self._try_generate(batch)
			if error and len(batch) > 1:
				# long prompts may not fit as a batch; retry them one at a time
				for slot in batch:
					slot_error = self._try_generate([slot])
					if slot_error:
						slot["error"] = slot_error
			elif error:
				batch[0]["error"] = error
			for slot in batch:
				slot["done"].set()

	def _generate(self, batch):
		texts = [
			self.tokenizer.apply_chat_template(
				slot["messages"], tokenize=False, add_generation_prompt=True,
				enable_thinking=not DISABLE_THINKING,
			)
			for slot in batch
		]
		inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
		with self.torch.no_grad():
			out = self.model.generate(
				**inputs, max_new_tokens=MAX_TOKENS, do_sample=False,
				temperature=None, top_p=None, top_k=None,
				pad_token_id=self.tokenizer.pad_token_id,
			)
		new_tokens = out[:, inputs["input_ids"].shape[1]:]
		for slot, mask, tokens in zip(batch, inputs["attention_mask"], new_tokens):
			n_out = int((tokens != self.tokenizer.pad_token_id).sum())
			slot["output"] = self.tokenizer.decode(tokens, skip_special_tokens=True)
			slot["meta"] = {
				"served_model": self.model_name,
				"model_revision": self.revision,
				"finish_reason": "length" if n_out >= MAX_TOKENS else "stop",
				"prompt_tokens": int(mask.sum()),
				"completion_tokens": n_out,
			}


_hf = None


def _complete_hf(messages, model, temperature):
	global _hf
	with _lock:
		if _hf is None:
			_hf = _HFBatcher(model)
	assert _hf.model_name == model, "hf backend serves one model per process"
	# TrialGPT always uses temperature 0, i.e. greedy decoding
	assert temperature == 0, "hf backend only implements greedy decoding"
	return _hf.submit(messages)


def chat(messages, model, temperature=0):
	"""Returns the message content of one chat completion (cached)."""
	params = {"temperature": temperature, "seed": SEED, "max_tokens": MAX_TOKENS}
	key = hashlib.sha256(
		json.dumps([BACKEND, model, messages, params], sort_keys=True).encode()
	).hexdigest()

	with _lock:
		cache = _load_cache()
		if key in cache:
			return cache[key]

	complete = _complete_hf if BACKEND == "hf" else _complete_api
	output, meta = complete(messages, model, temperature)

	with _lock:
		_cache[key] = output
		_append(CACHE_PATH, {"key": key, "output": output})
		_append(LOG_PATH, {
			"key": key,
			"backend": BACKEND,
			"model": model,
			"date": datetime.datetime.now().isoformat(),
			"seed": SEED,
			"temperature": temperature,
			**meta,
		})

	return output


def parse_json(text):
	"""Parse a JSON dict from model output.

	The original code only did text.strip("`").strip("json"); smaller open models
	often add <think> blocks or prose around the JSON, so we fall back to the
	outermost {...} span. Raises ValueError if nothing parses.
	"""
	text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
	try:
		return json.loads(text.strip("`").strip("json"))
	except json.JSONDecodeError:
		pass
	start, end = text.find("{"), text.rfind("}")
	if start != -1 and end > start:
		return json.loads(text[start:end + 1])
	raise ValueError("no JSON object in model output")


def safe_name(model):
	"""Model names like Qwen/Qwen3-8B cannot be used directly in file names."""
	return model.replace("/", "_")
