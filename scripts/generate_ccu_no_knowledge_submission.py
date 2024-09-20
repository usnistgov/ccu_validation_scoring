#! /usr/bin/env python
"""
A python script to generate a random submission. Currently, it only works
for norms and emotions.
"""
import os, glob
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from CCU_validation_scoring.preprocess_reference import *
import ccu_ref_analysis
import argparse

logger = logging.getLogger('GENERATING')
silence_string = "noann"

def PosNormal(mean):
  x = np.random.normal(mean,1)
  return(x if x>0 else PosNormal(mean))

def generate_submission_file(file_id, df, output_submission, index_df, nan_label):

	record_dict = {"file_id": file_id, "is_processed": nan_label, "message": "", "file_path": "{}.tab".format(file_id)}
	record_df = pd.DataFrame(record_dict, index=[0])
	new_index_df = pd.concat([index_df, record_df], ignore_index = True)
	df.to_csv(os.path.join(output_submission, "{}.tab".format(file_id)), sep = "\t", index = None)

	return new_index_df

def generate_submission_dir(output_dir, task, reference_dir):

	phase = "P1"
	task_number = "TA1"
	task_map = {"emotions": "ED"}
	task_label = task_map[task]
	team = "fake"
	dataset = os.path.basename(reference_dir)
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

def write_submission_record(stats, genre, length, i, j, task_column):

	Mean = stats.loc[(stats["class"] == str(j)), "mean"].values[0]
	
	if genre == "text":
		duration = round(PosNormal(Mean))
		new_end = length - duration
		start_time = np.random.randint(low=0,high=new_end)	
	else:
		duration = PosNormal(Mean)
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

def generate_no_knowledge_submission(task, reference_dir, scoring_index_file, output_dir):

	try:
		scoring_index = pd.read_csv(scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:GENERATING:{} is not a valid scoring index file'.format(scoring_index_file))
		exit(1)

	file_info = pd.read_csv(os.path.join(reference_dir,"docs","file_info.tab"), sep = "\t")
	index_df = file_info.merge(scoring_index, left_on = "file_uid", right_on = "file_id")
	index_df = index_df[["file_id", "type", "length"]]
	index_df.drop_duplicates(inplace = True)

	data_file = os.path.join(reference_dir,"data","{}.tab".format(task))
	data_df = read_dedupe_file(data_file)
	data_df = data_df[~data_df.isin(['EMPTY_TBD']).any(axis=1)]
	segment_file = os.path.join(reference_dir,"docs","segments.tab")
	segment_df = read_dedupe_file(segment_file)
	reference_df = data_df.merge(segment_df.merge(index_df))

	output_submission = generate_submission_dir(output_dir, task, reference_dir)

	index_df = pd.DataFrame(columns=["file_id", "is_processed", "message", "file_path"])

	task_column = task.replace("s","")

	for i in sorted(list(scoring_index["file_id"].unique())):
		sub_ref = reference_df.loc[reference_df["file_id"] == i]
		submission_df = pd.DataFrame(columns=["file_id",task_column,"start","end","llr"])
		for j in sorted(list(sub_ref["segment_id"].unique())):
			start_time = sub_ref.loc[sub_ref["segment_id"] == j, "start"].values[0]
			end_time = sub_ref.loc[sub_ref["segment_id"] == j, "end"].values[0]
			for emotion in ["anticipation", "fear", "joy", "sadness", "disgust", "anger", "trust", "surprise"]:
				record = pd.DataFrame({"file_id": i, task_column: emotion, "start": start_time,"end": end_time,"llr": 0.5}, index=[0])
				if submission_df.shape[0] == 0:
					submission_df = record
				else:
					submission_df = pd.concat([submission_df,record], ignore_index = True)
		type = file_info.loc[file_info["file_uid"] == i, "type"].values[0]
		if type == "text":
			submission_df['start'] = submission_df['start'].astype(int)
			submission_df['end'] = submission_df['end'].astype(int)
		index_df = generate_submission_file(i, submission_df, output_submission, index_df, "True")
	
	index_df.to_csv(os.path.join(output_submission, "system_output.index.tab"), sep = "\t", index = None)


def main():
	parser = argparse.ArgumentParser(description='Generate a no_knowledge emotion submission')
	parser.add_argument('-t', '--task', choices=['emotions'], required=True, help = 'norms, emotions')
	parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
	parser.add_argument('-i','--scoring-index-file', type=str, required=True, help='Use to filter file from scoring (REF)')
	parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory")

	args = parser.parse_args()

	generate_no_knowledge_submission(args.task, args.reference_dir, args.scoring_index_file, args.output_dir)

if __name__ == '__main__':
	main()