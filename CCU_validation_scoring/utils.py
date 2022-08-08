import os
import pandas as pd
import numpy as np
from .preprocess_reference import *

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

def mapping_known_hidden_norm(mapping_dir, hyp):
	"""
	Extract mapping info from mapping file and then modify old hyp by changing old system norm to new hidden reference norm
	"""
	mapping_file = os.path.join(mapping_dir, "nd.map.tab")
	mapping_df = pd.read_csv(mapping_file, dtype="object", sep = "\t")
	new_hyp = mapping_df.merge(hyp, left_on='sys_norm', right_on='Class')
	new_hyp = new_hyp[["file_id","start","end","status","llr","ref_norm"]]
	new_hyp.rename(columns={"ref_norm": "Class"}, inplace=True)
	return new_hyp

def extract_df(df, file_id):

	partial_df = df[df['file_id']==file_id]
	sorted_df = partial_df.sort_values(by=['start','end'])

	return sorted_df

def change_class_type(df, class_type):

    df["Class"] = df[class_type]
    df.drop(columns=[class_type],inplace=True)

    return df
