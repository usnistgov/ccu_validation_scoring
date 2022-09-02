import os
import re
import logging
import pandas as pd
import numpy as np

silence_string = "nospeech"

def tad_add_noscore_region(ref,hyp):
	""" 
	Convert nospeech class into NO_SCORE_REGION in ref and remove nospeech class in hyp
	"""    
	gtnan = ref[ref.Class == silence_string]
	gtnanl = len(gtnan)
	if gtnanl > 0:
		# logger.warning("Reference contains {} no-score regions.".format(gtnanl))
		ref.loc[ref.Class == silence_string, "Class"]= "NO_SCORE_REGION"

	prednan = hyp[hyp.Class == silence_string]
	prednanl = len(prednan)
	if prednanl > 0:
		logger = logging.getLogger('SCORING')
		logger.warning("Invalid or NaN Class in system-output detected. Dropping {} entries".format(prednanl))
		hyp.drop(hyp[hyp['Class'] == silence_string].index, inplace = True)
		hyp.drop(hyp[hyp.Class.isna()].index, inplace = True)

def ap_interp(prec, rec):
	"""Interpolated AP - Based on VOCdevkit from VOC 2011.
	"""
	mprec, mrec, idx = ap_interp_pr(prec, rec)
	ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
	return ap

def ap_interp_pr(prec, rec):
	"""Return Interpolated P/R curve - Based on VOCdevkit from VOC 2011.
	"""
	mprec = np.hstack([[0], prec, [0]])
	mrec = np.hstack([[0], rec, [1]])
	for i in range(len(mprec) - 1)[::-1]:
		mprec[i] = max(mprec[i], mprec[i + 1])
	idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
	return mprec, mrec, idx

def ensure_output_dir(output_dir):
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

def load_list(fn):
	try:
		fh = open(fn, "r")
		entries = fh.read()
		raw_list = entries.split("\n")
		return (list(filter(None, raw_list)))
	except IOError:
		print("File not found: '{}'".format(fn))
		exit(1)
				
def concatenate_submission_file(subm_dir, task):
	"""
	read all submission files in submission dir and concat into a global dataframe
	"""

	index_file_path = os.path.join(subm_dir, "system_output.index.tab")
	index_df = pd.read_csv(index_file_path, sep='\t')
	subm_file_paths = index_df["file_path"][index_df["is_processed"] == True]
	
	submission_dfs = pd.DataFrame()

	for subm_file_path in subm_file_paths:
		if subm_dir in subm_file_path:
			submission_df = pd.read_csv(subm_file_path, dtype={'norm': object}, sep='\t')
		else:
			submission_df = pd.read_csv(os.path.join(subm_dir,subm_file_path), dtype={'norm': object}, sep='\t')

		submission_dfs = pd.concat([submission_dfs, submission_df])

	submission_dfs.drop_duplicates(inplace = True)
	submission_dfs = submission_dfs.reset_index(drop=True)
	
	new_submission_dfs = change_class_type(submission_dfs, convert_task_column(task))

	return new_submission_dfs

def convert_task_column(task):

	if task == "norms" or task == "emotions":
		column_name = task.replace("s","")
	elif task == "changepoint":
		column_name = "timestamp"
	else:
		column_name = task

	return column_name

def add_type_column(ref, hyp):

	ref_type = ref[["file_id","type"]]
	ref_type_uniq = ref_type.drop_duplicates()
	hyp_type = hyp.merge(ref_type_uniq)

	return hyp_type

def extract_df(df, file_id):

	partial_df = df[df['file_id']==file_id]
	sorted_df = partial_df.sort_values(by=['start','end'])

	return sorted_df

def change_class_type(df, class_type):

	new_df = df.rename(columns = {class_type: "Class"})

	return new_df

def replace_hyp_norm_mapping(sub_mapping_df, hyp, act):

	sys_norm_list = list(sub_mapping_df.sys_norm)
	sub_hyp = hyp[hyp.Class.isin(sys_norm_list)]
	new_sub_hyp = sub_mapping_df.merge(sub_hyp, left_on='sys_norm', right_on='Class')
	new_sub_hyp = new_sub_hyp[["file_id","ref_norm","start","end","status","llr"]]
	new_sub_hyp.rename(columns={"ref_norm": "Class"}, inplace=True)
	new_sub_hyp.drop_duplicates(inplace = True)
	final_sub_hyp = pd.concat([new_sub_hyp, hyp.loc[(hyp.Class == act)]])

	return final_sub_hyp