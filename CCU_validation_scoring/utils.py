import os
import re
import logging
import pandas as pd
import numpy as np
from .preprocess_reference import *

def tad_add_noscore_region(ref,hyp):
	""" 
	Convert nospeech class into NO_SCORE_REGION in ref and remove nospeech class in hyp
	"""    
	gtnan = ref[ref.Class == "nospeech"]
	gtnanl = len(gtnan)
	if gtnanl > 0:
		# logger.warning("Reference contains {} no-score regions.".format(gtnanl))
		ref.loc[ref.Class == "nospeech", "Class"]= "NO_SCORE_REGION"

	prednan = hyp[hyp.Class == "nospeech"]
	prednanl = len(prednan)
	if prednanl > 0:
		logger = logging.getLogger('SCORING')
		logger.warning("NaN Class in system-output detected. Dropping {} NaN entries".format(prednanl))
		hyp.drop(hyp[hyp['Class'] == "nospeech"].index, inplace = True)

def remove_out_of_scope_activities(ref,hyp):
	""" 
	If there are any Class which are out of scope or NA, whole entry is
	removed.    

	"""
	# ref.Class will already include NO_SCORE_REGION Class 
	hyp.drop(hyp[~hyp.Class.isin(ref.Class.unique())].index, inplace = True)
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

	new_df = df.rename(columns = {class_type: "Class"})

	return new_df
