__author__ = "qiao"

"""
Running the TrialGPT matching for three cohorts (sigir, TREC 2021, TREC 2022).

Team Zenith additions: patient subsets (--n_patients / --patient_ids),
--max_trials_per_label for smoke tests, and concurrent requests (--workers).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from nltk.tokenize import sent_tokenize
import os
import sys
import threading

from trialguide.baselines.trialgpt.matching import trialgpt_matching
from trialguide.llm.client import safe_name

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("corpus")
	parser.add_argument("model")
	parser.add_argument("--n_patients", type=int, default=None, help="first N patients only")
	parser.add_argument("--patient_ids", default=None, help="file with one patient id per line")
	parser.add_argument("--max_trials_per_label", type=int, default=None)
	parser.add_argument("--workers", type=int, default=16)
	parser.add_argument("--tag", default="", help="suffix for the output file")
	parser.add_argument("--data_dir", default=os.getenv("TRIALGPT_DATA", "data/raw/trialgpt"),
		help="folder with {corpus}/retrieved_trials.json etc. (TrialGPT's dataset/ folder)")
	parser.add_argument("--out_dir", default="outputs/trialgpt")
	args = parser.parse_args()

	corpus = args.corpus
	model = args.model

	dataset = json.load(open(f"{args.data_dir}/{corpus}/retrieved_trials.json"))
	if args.patient_ids:
		keep = set(open(args.patient_ids).read().split())
		dataset = [inst for inst in dataset if inst["patient_id"] in keep]
	if args.n_patients:
		dataset = dataset[:args.n_patients]

	os.makedirs(args.out_dir, exist_ok=True)
	output_path = f"{args.out_dir}/matching_results_{corpus}_{safe_name(model)}{args.tag}.json"

	# Dict{Str(patient_id): Dict{Str(label): Dict{Str(trial_id): Str(output)}}}
	if os.path.exists(output_path):
		output = json.load(open(output_path))
	else:
		output = {}

	jobs = []
	for instance in dataset:
		# Dict{'patient': Str(patient), '0': Str(NCTID), ...}
		patient_id = instance["patient_id"]
		patient = instance["patient"]
		sents = sent_tokenize(patient)
		sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
		sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
		patient = "\n".join(sents)

		# initialize the patient id in the output
		if patient_id not in output:
			output[patient_id] = {"0": {}, "1": {}, "2": {}}

		for label in ["2", "1", "0"]:
			if label not in instance: continue

			for trial in instance[label][:args.max_trials_per_label]:
				# already calculated and cached (retry if a side failed to parse)
				cached = output[patient_id][label].get(trial["NCTID"])
				if isinstance(cached, dict) and all(isinstance(cached.get(side), dict) for side in ["inclusion", "exclusion"]):
					continue
				jobs.append((patient_id, label, trial, patient))

	print(f"{len(dataset)} patients, {len(jobs)} patient-trial pairs to match")

	lock = threading.Lock()

	def save():
		with open(output_path, "w") as f:
			json.dump(output, f, indent=4)

	failed = 0
	with ThreadPoolExecutor(args.workers) as pool:
		futures = {
			pool.submit(trialgpt_matching, trial, patient, model): (patient_id, label, trial["NCTID"])
			for patient_id, label, trial, patient in jobs
		}
		for n, future in enumerate(as_completed(futures), 1):
			patient_id, label, trial_id = futures[future]
			# in case anything goes wrong (e.g., API calling errors)
			try:
				results = future.result()
			except Exception as e:
				print(patient_id, trial_id, e, flush=True)
				failed += 1
				continue
			with lock:
				output[patient_id][label][trial_id] = results
				if n % 50 == 0:
					save()
					print(f"{n}/{len(jobs)} done", flush=True)

	save()

	# fail loudly so the pipeline stops instead of evaluating partial results
	if failed:
		sys.exit(f"{failed}/{len(jobs)} pairs failed; rerun to retry them (finished calls are cached)")
