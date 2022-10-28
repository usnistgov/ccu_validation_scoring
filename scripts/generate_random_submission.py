import os
import numpy as np
import pandas as pd
import math
from datetime import datetime
import logging
from CCU_validation_scoring.preprocess_reference import *
import ccu_ref_analysis
import argparse

logger = logging.getLogger('GENERATING')
silence_string = "noann"

def generate_submission_file(file_id, df, output_submission, index_df):

	record_dict = {"file_id": file_id, "is_processed": "True", "message": "", "file_path": "{}.tab".format(file_id)}
	record_df = pd.DataFrame(record_dict, index=[0])
	new_index_df = pd.concat([index_df, record_df], ignore_index = True)
	df.to_csv(os.path.join(output_submission, "{}.tab".format(file_id)), sep = "\t", index = None)

	return new_index_df

def generate_submission_dir(output_dir, task):

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

	return output_submission

def write_submission_record(stats_ref, i, j, task_column):

	Mean = stats_ref.loc[(stats_ref["file_id"] == i) & (stats_ref["class"] == j), "mean"].values[0]
	Std_dev = stats_ref.loc[(stats_ref["file_id"] == i) & (stats_ref["class"] == j), "stdev"].values[0]
	genre = stats_ref.loc[(stats_ref["file_id"] == i) & (stats_ref["class"] == j), "genre"].values[0]
	length = list(stats_ref.loc[stats_ref["file_id"] == i, "length"])[0]
	
	if genre == "text":
		duration = round(np.random.normal(loc=Mean, scale=Std_dev))
		new_end = length - duration
		start_time = np.random.randint(low=0,high=new_end)	
	else:
		duration = np.random.normal(loc=Mean, scale=Std_dev)
		new_end = length - duration
		start_time = np.random.uniform(low=0,high=new_end)

	end_time = start_time + duration

	llr = np.random.normal(loc=0, scale=1)
	status = np.random.choice(["adhere","violate"])

	if task_column == "norm":
		status = np.random.choice(["adhere","violate"])
		submission_record = pd.DataFrame({"file_id": i,task_column: j,"start": start_time,"end": end_time,"status": status,"llr": llr}, index=[0])
	if task_column == "emotion":
		submission_record = pd.DataFrame({"file_id": i,task_column: j,"start": start_time,"end": end_time,"llr": llr}, index=[0])

	return submission_record

def generate_random_submission(task, reference_dir, scoring_index_file, output_dir):

	try:
		scoring_index = pd.read_csv(scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:GENERATING:{} is not a valid scoring index file'.format(scoring_index_file))
		exit(1)

	ref = preprocess_reference_dir(reference_dir, scoring_index, task)
	stats = ccu_ref_analysis.compute_stats(ref)
	stats_pruned = stats[["class","genre","mean","stdev"]]
	stats_pruned = stats_pruned.loc[stats_pruned["class"] != silence_string]
	stats_pruned = stats_pruned.fillna(0)

	ref.rename(columns={'Class':'class', 'type':'genre'}, inplace=True)

	stats_ref = ref.merge(stats_pruned, on=["class","genre"])

	output_submission = generate_submission_dir(output_dir, task)

	index_df = pd.DataFrame(columns=["file_id", "is_processed", "message", "file_path"])

	for i in sorted(list(set(stats_ref.file_id))):
		Class = stats_ref.loc[stats_ref["file_id"] == i, "class"]
		task_column = task.replace("s","")
		if task_column == "norm":
			submission_df = pd.DataFrame(columns=["file_id",task_column,"start","end","status","llr"])
		if task_column == "emotion":
			submission_df = pd.DataFrame(columns=["file_id",task_column,"start","end","llr"])
		for j in sorted(list(set(Class))):
			for time in range(2):
				record = write_submission_record(stats_ref, i, j, task_column)
				submission_df = pd.concat([submission_df,record], ignore_index = True)

		index_df = generate_submission_file(i, submission_df, output_submission, index_df)
	
	index_df.to_csv(os.path.join(output_submission, "system_output.index.tab"), sep = "\t", index = None)


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