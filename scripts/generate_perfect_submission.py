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
import argparse

logger = logging.getLogger('GENERATING')
silence_string = "noann"

def generate_submission_file(file_id, df, output_submission, index_df, nan_label):

	record_dict = {"file_id": file_id, "is_processed": nan_label, "message": "", "file_path": "{}.tab".format(file_id)}
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
	dataset = "LDC2022E22-V1"
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

def generate_perfect_submission(task, submission_format, reference_dir, scoring_index_file, output_dir, text_gap, time_gap, merge_label, minimum_vote_agreement, fix_ref_status_conflict = None):

	try:
		scoring_index = pd.read_csv(scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:GENERATING:{} is not a valid scoring index file'.format(scoring_index_file))
		exit(1)

	ref = preprocess_reference_dir(reference_dir, scoring_index, task, text_gap, time_gap, merge_label, fix_ref_status_conflict_label=fix_ref_status_conflict, minimum_vote_agreement = minimum_vote_agreement)
	if (task == 'norms'):
		ref.loc[ref["status"] == "adhere,violate", "status"] = "adhere"

	output_submission = generate_submission_dir(output_dir, task)

	index_df = pd.DataFrame(columns=["file_id", "is_processed", "message", "file_path"])

	system_input_index_file_path = os.path.join(reference_dir, "index_files", "*system_input.index.tab")
	system_input_index_df = pd.read_csv(glob.glob(system_input_index_file_path)[0], sep = "\t")

	file_info_df = pd.read_csv(os.path.join(reference_dir, "docs", "file_info.tab"), sep='\t')

	for i in sorted(list(system_input_index_df["file_id"].unique())):

		task_column = task.replace("s","")
		
		sub_ref = ref.loc[ref["file_id"] == i]
		sub_ref_ann = sub_ref.loc[sub_ref["Class"] != "noann"]
		if (task == 'norms'):
			if (submission_format == "open"):
				segment_df = pd.read_csv(os.path.join(reference_dir,"docs","segments.tab"), sep = "\t")
				sub_ref_ann = sub_ref_ann.merge(segment_df, on=["file_id","start","end"])
				filter_sub_ref_ann = sub_ref_ann[["file_id","segment_id","Class","status"]]
			else:
				filter_sub_ref_ann = sub_ref_ann[["file_id","Class","start","end","status"]]
		else:
			filter_sub_ref_ann = sub_ref_ann[["file_id","Class","start","end"]]

		rename_filter_sub_ref_ann = filter_sub_ref_ann.rename(columns = {"Class": task_column})
		type = file_info_df.loc[file_info_df["file_uid"] == i, "type"].values[0]
		if type == "text":
			rename_filter_sub_ref_ann['start'] = filter_sub_ref_ann['start'].astype(int)
			rename_filter_sub_ref_ann['end'] = filter_sub_ref_ann['end'].astype(int)
		rename_filter_sub_ref_ann["llr"] = 0.5  
		index_df = generate_submission_file(i, rename_filter_sub_ref_ann, output_submission, index_df, "True")
	index_df.to_csv(os.path.join(output_submission, "system_output.index.tab"), sep = "\t", index = None)

def main():
	parser = argparse.ArgumentParser(description='Generate a random norm/emotion submission')
	parser.add_argument('-t', '--task', choices=['norms', 'emotions'], required=True, help = 'norms, emotions')
	parser.add_argument('-sf','--submission-format', type=str, default="regular", choices=['regular','open'], help='choose submission format, regular is for CCU evaluation while open is for OpenCCU')
	parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
	parser.add_argument("-xR", "--merge_ref_text_gap", type=str, required=False, help="merge reference text gap character")
	parser.add_argument("-aR", "--merge_ref_time_gap", type=str, required=False, help="merge reference time gap second")
	parser.add_argument("-vR", "--merge_ref_label", type=str, choices=['class', 'class-status'], required=False, help="choose class or class-status to define how to handle the adhere/violate labels for the reference norm instances merging. class is to use the class label only (ignoring status) to merge and class-status is to use the class and status label to merge")
	parser.add_argument("-f", "--fix_ref_status_conflict", action='store_true', help="set reference annotation to noann when there are the same annotations but different status")
	parser.add_argument("-mv", "--minimum_vote_agreement", type=int, default=2, required=False, help="Set the mimimum agreement for voting between annotators. Default is 2 agreeing annotators per segment.")
	parser.add_argument('-i','--scoring-index-file', type=str, required=True, help='Use to filter file from scoring (REF)')
	parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory")

	args = parser.parse_args()

	if args.merge_ref_text_gap:
		merge_ref_text_gap = int(args.merge_ref_text_gap)
	else:
		merge_ref_text_gap = None

	if args.merge_ref_time_gap:
		merge_ref_time_gap = float(args.merge_ref_time_gap)
	else:
		merge_ref_time_gap = None

	generate_perfect_submission(args.task, args.submission_format, args.reference_dir, args.scoring_index_file, args.output_dir, merge_ref_text_gap, merge_ref_time_gap, args.merge_ref_label, args.minimum_vote_agreement, args.fix_ref_status_conflict)

if __name__ == '__main__':
	main()