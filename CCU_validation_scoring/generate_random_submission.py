import os
import numpy as np
import pandas as pd
import math
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

def generate_random_submission(task, reference_dir, scoring_index_file):

	try:
		scoring_index = pd.read_csv(scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:GENERATING:{} is not a valid scoring index file'.format(scoring_index_file))
		exit(1)

	ref = preprocess_reference_dir(reference_dir, scoring_index, task)
	ref["duration"] = ref["end"] - ref["start"]

	for i in list(set(ref.file_id)):

		length = list(ref.loc[ref["file_id"] == i, "length"])[0]
		start_time = np.random.uniform(low=0,high=length)

		durations = list(ref.loc[ref["file_id"] == i, "duration"])
		mean, std_dev = mean_stdev(durations)
		duration = np.random.normal(loc=mean, scale=std_dev)
		end_time = start_time + duration

		llr = np.random.normal(loc=0, scale=1)

		print(i, start_time, mean, std_dev)
	
 
def main():
	parser = argparse.ArgumentParser(description='Generate a random submission')
	parser.add_argument('-t','--task', type=str, required=True, help = 'norms, emotions, valence_continuous, arousal_continuous or changepoint')
	parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
	parser.add_argument('-i','--scoring-index-file', type=str, required=True, help='Use to filter file from scoring (REF)')

	args = parser.parse_args()
	generate_random_submission(args.task, args.reference_dir, args.scoring_index_file)

if __name__ == '__main__':
	main()