import os
import numpy as np
import pandas as pd
import math
from datetime import datetime
import logging
from CCU_validation_scoring.preprocess_reference import *
import argparse

logger = logging.getLogger('GENERATING')
 
def mean_stdev(data):

  n = len(data)
  mean = sum(data) / n
  deviations = [(x - mean) ** 2 for x in data]
  variance = sum(deviations) / n
  std_dev = math.sqrt(variance)
  
  return mean, std_dev

def generate_submission_file(file_id, df, output_submission, index_df):

	record_df = {"file_id": file_id, "is_processed": "True", "message": "", "file_path": "{}.tab".format(file_id)}
	new_index_df = index_df.append(record_df, ignore_index = True)
	df.to_csv(os.path.join(output_submission, "{}.tab".format(file_id)))

	return new_index_df

def generate_random_submission(task, reference_dir, scoring_index_file, output_dir):

	try:
		scoring_index = pd.read_csv(scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:GENERATING:{} is not a valid scoring index file'.format(scoring_index_file))
		exit(1)

	ref = preprocess_reference_dir(reference_dir, scoring_index, task)
	ref["duration"] = ref["end"] - ref["start"]

	phase = "P1"
	task_number = "TA1"
	task_map = {"norms": "ND", "emotions": "ED", "valence_continuous": "VD", "arousal_continuous": "AD", "changepoint": "CD"}
	task_label = task_map[task]
	team = "fake"
	dataset = "LDC2022R17-V1"
	now = datetime.now()
	date = now.strftime("%Y%m%d")
	time = now.strftime("%H%M%S")

	submission_name = "CCU_{}_{}_{}_{}_{}_{}_{}".format(phase, task_number, task_label, team, dataset, date, time)

	output_submission = os.path.join(output_dir, submission_name)

	if not os.path.exists(output_submission):
		print('Creating {}'.format(output_submission))
		os.makedirs( os.path.dirname(output_submission + '/'), mode=0o777, exist_ok=False)
	else:
		print('Directory {} already exists, delete it manualy it'.format(output_submission))

	for i in list(set(ref.file_id)):

		length = list(ref.loc[ref["file_id"] == i, "length"])[0]
		start_time = np.random.uniform(low=0,high=length)

		durations = list(ref.loc[ref["file_id"] == i, "duration"])
		mean, std_dev = mean_stdev(durations)
		duration = np.random.normal(loc=mean, scale=std_dev)
		end_time = start_time + duration

		llr = np.random.normal(loc=0, scale=1)

		index_df = pd.DataFrame(columns=["file_id", "is_processed", "message", "file_path"])
		index_df = generate_submission_file(i, df, output_submission, index_df, task)
	
	index_df.to_csv(os.path.join(output_submission, "system_output.index.tab"))


def main():
	parser = argparse.ArgumentParser(description='Generate a random submission')
	parser.add_argument('-t','--task', type=str, required=True, help='norms, emotions, valence_continuous, arousal_continuous or changepoint')
	parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
	parser.add_argument('-i','--scoring-index-file', type=str, required=True, help='Use to filter file from scoring (REF)')
	parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory")

	args = parser.parse_args()
	generate_random_submission(args.task, args.reference_dir, args.scoring_index_file, args.output_dir)

if __name__ == '__main__':
	main()