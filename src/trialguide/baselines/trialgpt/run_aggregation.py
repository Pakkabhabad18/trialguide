__author__ = "qiao"

"""
Using GPT to aggregate the scores by itself.

Team Zenith additions: concurrent requests (--workers) and safe output names.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from nltk.tokenize import sent_tokenize
import os
import sys

from trialguide.baselines.trialgpt.ranking import trialgpt_aggregation
from trialguide.llm.client import safe_name

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("corpus")
	parser.add_argument("model")
	# the path of the matching results
	parser.add_argument("matching_results_path")
	parser.add_argument("--workers", type=int, default=16)
	parser.add_argument("--tag", default="", help="suffix for the output file")
	parser.add_argument("--data_dir", default=os.getenv("TRIALGPT_DATA", "data/trialgpt"),
		help="folder with {corpus}/retrieved_trials.json etc. (TrialGPT's dataset/ folder)")
	parser.add_argument("--out_dir", default="outputs/trialgpt")
	args = parser.parse_args()

	corpus = args.corpus
	model = args.model
	results = json.load(open(args.matching_results_path))

	# loading the trial2info dict. retrieved_trials.json carries every field the
	# aggregation prompt needs for each candidate (the same dicts the matching step
	# saw), so we use it instead of the 1.2 GB dataset/trial_info.json
	trial2info = {
		trial["NCTID"]: trial
		for inst in json.load(open(f"{args.data_dir}/{corpus}/retrieved_trials.json"))
		for label in ["0", "1", "2"]
		for trial in inst.get(label, [])
	}

	# loading the patient info
	# (read directly instead of beir's GenericDataLoader, which needs extra deps)
	with open(f"{args.data_dir}/{corpus}/queries.jsonl") as f:
		queries = {entry["_id"]: entry["text"] for entry in map(json.loads, f)}

	# output file path
	os.makedirs(args.out_dir, exist_ok=True)
	output_path = f"{args.out_dir}/aggregation_results_{corpus}_{safe_name(model)}{args.tag}.json"

	if os.path.exists(output_path):
		output = json.load(open(output_path))
	else:
		output = {}

	jobs = []
	# patient-level
	for patient_id, info in results.items():
		# get the patient note
		patient = queries[patient_id]
		sents = sent_tokenize(patient)
		sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
		sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
		patient = "\n".join(sents)

		if patient_id not in output:
			output[patient_id] = {}

		# label-level, 3 label / patient
		for label, trials in info.items():

			# trial-level
			for trial_id, trial_results in trials.items():
				# already cached results (a "matching result error" placeholder is
				# retried, since a rerun of matching may have fixed that pair)
				if isinstance(output[patient_id].get(trial_id), dict):
					continue

				# also skip pairs where one side's model output did not parse
				if type(trial_results) is not dict or not all(
					isinstance(trial_results.get(side), dict) for side in ["inclusion", "exclusion"]
				):
					output[patient_id][trial_id] = "matching result error"
					continue

				jobs.append((patient_id, trial_id, patient, trial_results, trial2info[trial_id]))

	print(f"{len(jobs)} patient-trial pairs to aggregate")

	def save():
		with open(output_path, "w") as f:
			json.dump(output, f, indent=4)

	failed = 0
	with ThreadPoolExecutor(args.workers) as pool:
		futures = {
			pool.submit(trialgpt_aggregation, patient, trial_results, trial_info, model): (patient_id, trial_id)
			for patient_id, trial_id, patient, trial_results, trial_info in jobs
		}
		for n, future in enumerate(as_completed(futures), 1):
			patient_id, trial_id = futures[future]
			try:
				output[patient_id][trial_id] = future.result()
			except Exception as e:
				print(patient_id, trial_id, e, flush=True)
				failed += 1
				continue
			if n % 50 == 0:
				save()
				print(f"{n}/{len(jobs)} done", flush=True)

	save()

	# fail loudly so the pipeline stops instead of evaluating partial results
	if failed:
		sys.exit(f"{failed}/{len(jobs)} pairs failed; rerun to retry them (finished calls are cached)")
