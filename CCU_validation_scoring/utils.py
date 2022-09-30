import os
import re
import logging
import pandas as pd
import numpy as np

silence_string = "noann"

def is_float(value):
	try: 
		float(value)
		return True
	except ValueError:
		return False
		
def tad_add_noscore_region(ref,hyp):
	""" 
	Convert noann class into NO_SCORE_REGION in ref and remove noann class in hyp
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

def get_unique_items_in_array(file_id_array):
	"""
		Extract unique items from an array and return in array format
	 
		Parameters
		----------
		file_id_array : array
 
		Returns
		-------
		list
	"""
	return list(set(file_id_array))

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

		if task == "norms" or task == "emotions":
			submission_df_sorted = submission_df.sort_values(by=['start','end'])
			submission_dfs = pd.concat([submission_dfs, submission_df_sorted])
		if task == "valence_continuous" or task == "arousal_continuous":
			submission_df_sorted = submission_df.sort_values(by=['start','end'])
			submission_df_filled = fill_epsilon_submission(submission_df_sorted)
			submission_dfs = pd.concat([submission_dfs, submission_df_filled])
		if task == "changepoint":
			submission_dfs = pd.concat([submission_dfs, submission_df])

	submission_dfs.drop_duplicates(inplace = True)
	submission_dfs = submission_dfs.reset_index(drop=True)
	
	new_submission_dfs = change_class_type(submission_dfs, convert_task_column(task))

	return new_submission_dfs

def fill_epsilon_submission(sys):

	for i in range(sys.shape[0]-1):
		if sys.iloc[i]["end"] != sys.iloc[i+1]["start"] and abs(sys.iloc[i]["end"] - sys.iloc[i+1]["start"]) < 0.02:
			sys.iloc[i+1, sys.columns.get_loc('start')] = sys.iloc[i]["end"]
	
	return sys

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

def formatNumber(num):
  if num % 1 == 0:
    return str(int(num))
  else:
    return str(num)

def replace_hyp_norm_mapping(sub_mapping_df, hyp, act):

	sys_norm_list = list(sub_mapping_df.sys_norm)
	sub_hyp = hyp[hyp.Class.isin(sys_norm_list)]
	new_sub_hyp = sub_mapping_df.merge(sub_hyp, left_on='sys_norm', right_on='Class')
	new_sub_hyp = new_sub_hyp[["file_id","ref_norm","start","end","status","llr","type"]]
	new_sub_hyp.rename(columns={"ref_norm": "Class"}, inplace=True)
	new_sub_hyp.drop_duplicates(inplace = True)
	final_sub_hyp = pd.concat([new_sub_hyp, hyp.loc[(hyp.Class == act)]])

	return final_sub_hyp

def generate_alignment_file(ref, hyp, task):

	def categorise(row):
		if row['tp'] == 1 and row['fp'] == 0:
			return 'mapped'
		return 'unmapped'

	hyp["eval"] = hyp.apply(lambda row: categorise(row), axis=1)

	if task in ["norm","emotion"]:

		ref_format = ref.copy()
		ref_format['start'] = [formatNumber(x) for x in ref['start']]
		ref_format['end'] = [formatNumber(x) for x in ref['end']]
		ref_format['sort'] = [x for x in ref['start']]
		
		hyp_format = hyp.copy()
		hyp_format['start_ref'] = [formatNumber(x) for x in hyp['start_ref']]
		hyp_format['end_ref'] = [formatNumber(x) for x in hyp['end_ref']]
		hyp_format['start_hyp'] = [formatNumber(x) for x in hyp['start_hyp']]
		hyp_format['end_hyp'] = [formatNumber(x) for x in hyp['end_hyp']]
		hyp_format['sort'] = [x for x in hyp['start_hyp']]

		hyp_format["ref"] = "{start=" + hyp_format["start_ref"].astype(str) + ",end=" + hyp_format["end_ref"].astype(str) + "}"
		hyp_format["sys"] = "{start=" + hyp_format["start_hyp"].astype(str) + ",end=" + hyp_format["end_hyp"].astype(str) + "}"
		hyp_format["IoU_format"] = hyp_format["IoU"].apply(lambda x: "{:,.3f}".format(x))
		hyp_format["parameters"] = '{iou=' + hyp_format["IoU_format"] + '}'

		hyp_format.loc[hyp_format["eval"] == "unmapped", "ref"] = "{}"
		hyp_format.loc[hyp_format["eval"] == "unmapped", "parameters"] = "{}"

		hyp_format = hyp_format[["Class","file_id","eval","ref","sys","llr","parameters","sort"]]
		ref_new = ref_format.copy()
		ref_new["ref"] = "{start=" + ref_format["start"].astype(str) + ",end=" + ref_format["end"].astype(str) + "}"
		ref_new = ref_new[["file_id","Class","ref","sort"]]
		ref_new = ref_new.loc[~(ref_new["ref"].isin(hyp_format["ref"]))]
		ref_new["eval"] = "unmapped"
		ref_new["sys"] = "{}"
		ref_new["parameters"] = "{}"
		ref_new["llr"] = np.nan

		alignment = pd.concat([hyp_format, ref_new])
		alignment = alignment.rename(columns={'Class':'class'})
		print(alignment)
	if task == "changepoint":

		ref_format = ref.copy()
		ref_format['Class'] = [formatNumber(x) for x in ref['Class']]
		
		hyp_format = hyp.copy()
		hyp_format['Class_ref'] = [formatNumber(x) for x in hyp['Class_ref']]
		hyp_format['Class_hyp'] = [formatNumber(x) for x in hyp['Class_hyp']]

		hyp_format["ref"] = '{timestamp=' + hyp_format["Class_ref"].astype(str) + '}'
		hyp_format["sys"] = '{timestamp=' + hyp_format["Class_hyp"].astype(str) + '}'
		hyp_format["parameters"] = '{delta_cp=' + hyp_format["delta_cp"].astype(str) + '}'

		hyp_format.loc[hyp_format["eval"] == "unmapped", "ref"] = "{}"
		hyp_format.loc[hyp_format["eval"] == "unmapped", "parameters"] = "{}"
		hyp_format["Class"] = "cp"

		hyp_format = hyp_format[["Class","file_id","eval","ref","sys","llr","parameters"]]
		ref_new = ref_format.copy()
		ref_new["ref"] = '{timestamp=' + ref_format["Class"].astype(str) + '}'
		ref_new["Class"] = "cp"
		ref_new = ref_new[["file_id","Class","ref"]]
		ref_new = ref_new.loc[~(ref_new["ref"].isin(hyp_format["ref"]))]
		ref_new["eval"] = "unmapped"
		ref_new["sys"] = "{}"
		ref_new["parameters"] = "{}"
		ref_new["llr"] = np.nan

		alignment = pd.concat([hyp_format, ref_new])
		alignment = alignment.rename(columns={'Class':'class'})

	return alignment

def generate_all_fn_alignment_file(ref, task):
	
	if task in ["norm","emotion"]:

		ref_format = ref.copy()
		ref_format['start'] = [formatNumber(x) for x in ref['start']]
		ref_format['end'] = [formatNumber(x) for x in ref['end']]
		ref_format['sort'] = [x for x in ref['start']]

		ref_new = ref_format.loc[ref_format.Class.str.contains('NO_SCORE_REGION')==False].copy()
		ref_new["ref"] = "{start=" + ref_format["start"].astype(str) + ",end=" + ref_format["end"].astype(str) + "}"
		ref_new = ref_new[["file_id","Class","ref"]]
		ref_new["eval"] = "unmapped"
		ref_new["sys"] = "{}"
		ref_new["parameters"] = "{}"
		ref_new["llr"] = np.nan
		ref_new["sort"] = ref_format['sort']
		ref_new = ref_new.rename(columns={'Class':'class'})
		ref_new = ref_new[["class","file_id","eval","ref","sys","llr","parameters","sort"]]
	if task == "changepoint":

		ref_format = ref.copy()
		ref_format['Class'] = [formatNumber(x) for x in ref['Class']]

		ref_new = ref_format.loc[ref_format.Class != 'NO_SCORE_REGION'].copy()
		ref_new["ref"] = '{timestamp=' + ref_format['Class'].astype(str) + '}'
		ref_new = ref_new[["file_id","Class","ref"]]
		ref_new["eval"] = "unmapped"
		ref_new["sys"] = "{}"
		ref_new["parameters"] = "{}"
		ref_new["llr"] = np.nan
		ref_new["class"] = "cp"
		ref_new = ref_new[["class","file_id","eval","ref","sys","llr","parameters"]]

	return ref_new
