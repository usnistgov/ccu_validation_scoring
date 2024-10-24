import os
import fnmatch
import subprocess
import logging
import pandas as pd
import numpy as np
import scipy.stats as stats
import re
from matplotlib import pyplot as plt
from .aggregate import *

silence_string = "noann"

def is_float(value):
	"""
	Check if value is float type
	"""
	try: 
		float(value)
		return True
	except ValueError:
		return False
	
def read_dedupe_file(path):
	"""
	Read file and remove duplicate records 
	"""
	df = pd.read_csv(path, dtype={'norm': object}, sep = "\t")
	df.drop_duplicates(inplace = True)

	return df
		
def tad_add_noscore_region(ref,hyp):
	""" 
	Convert noann class into NO_SCORE_REGION in ref and remove noann class in hyp
	"""    
	gtnan = ref[ref.Class == silence_string]
	gtnanl = len(gtnan)
	if gtnanl > 0:
		# logger.warning("Reference contains {} no-score regions.".format(gtnanl))
		ref.loc[ref.Class == silence_string, "Class"]= "NO_SCORE_REGION"

	if hyp.shape[0] > 0:
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
	if os.path.isfile(output_dir):
		logger = logging.getLogger('SCORING')
		logger.error("The output directory '{}' is a file and exists. Please remove or specify another output directory.".format(output_dir))
		exit(1)
	if not os.path.isdir(output_dir):
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
	"""
	read norm/emotion list from file
	"""
	try:
		fh = open(fn, "r")
		entries = fh.read()
		raw_list = entries.split("\n")
		return (list(filter(None, raw_list)))
	except IOError:
		print("File not found: '{}'".format(fn))
		exit(1)

def check_scoring_index_out_of_scope(ref_dir, scoring_index, task):

	version_per_df = pd.read_csv(os.path.join(ref_dir,"docs/versions_per_file.tab"), sep = "\t")

	if task == "norms" or task == "emotions" or task == "changepoint":
		ann_file_list = set(version_per_df.loc[version_per_df["{}_count".format(task)] > 0, "file_id"])
	if task == "valence_continuous" or task == "arousal_continuous":
		ann_file_list = set(version_per_df.loc[version_per_df["valence_arousal_count"] > 0, "file_id"])
	
	scoring_index_file_list = set(scoring_index["file_id"])

	invalid_file = scoring_index_file_list - ann_file_list
	if invalid_file:
		logger = logging.getLogger('SCORING')
		logger.error("Additional file(s) '{}' have been found in scoring_index".format(invalid_file))
		exit(1)
				
def concatenate_submission_file(subm_dir, task):
	"""
	read all submission files in submission dir and concat into a global dataframe
	"""

	index_file_path = os.path.join(subm_dir, "system_output.index.tab")
	index_df = pd.read_csv(index_file_path, dtype={'message': object}, sep='\t')
	subm_file_paths = index_df["file_path"][index_df["is_processed"] == True]
	
	submission_dfs = pd.DataFrame()

	for subm_file_path in subm_file_paths:
		if "{}/".format(subm_dir) in subm_file_path:
			submission_df = pd.read_csv(subm_file_path, dtype={'norm': object}, sep='\t')
		else:
			submission_df = pd.read_csv(os.path.join(subm_dir,subm_file_path), dtype={'norm': object}, sep='\t')

		if task == "norms" or task == "emotions":
			submission_df_sorted = submission_df.sort_values(by=['start','end'])
			if submission_df_sorted.shape[0] > 0:
				submission_dfs = pd.concat([submission_dfs, submission_df_sorted])
		if task == "valence_continuous" or task == "arousal_continuous":
			submission_df_sorted = submission_df.sort_values(by=['start','end'])
			submission_df_filled = fill_epsilon_submission(submission_df_sorted, 0.02)
			if submission_df_filled.shape[0] > 0:
				submission_dfs = pd.concat([submission_dfs, submission_df_filled])
		if task == "changepoint":
			if submission_df.shape[0] > 0:
				submission_dfs = pd.concat([submission_dfs, submission_df])

	submission_dfs.drop_duplicates(inplace = True)
	submission_dfs = submission_dfs.reset_index(drop=True)

	if task == "valence_continuous" or task == "arousal_continuous":
		submission_dfs[task] = stats.zscore(np.array(submission_dfs[task]))

	new_submission_dfs = change_class_type(submission_dfs, convert_task_column(task))

	return new_submission_dfs

def concatenate_openccu_submission_file(subm_dir, ref_dir, task):
	"""
	read all submission files in submission dir and concat into a global dataframe
	"""
	segment_file = os.path.join(ref_dir,"docs","segments.tab")
	segment_df = read_dedupe_file(segment_file)
	index_file_path = os.path.join(subm_dir, "system_output.index.tab")
	index_df = pd.read_csv(index_file_path, dtype={'message': object}, sep='\t')
	subm_file_paths = index_df["file_path"][index_df["is_processed"] == True]
	
	submission_dfs = pd.DataFrame()

	for subm_file_path in subm_file_paths:
		if "{}/".format(subm_dir) in subm_file_path:
			submission_df = pd.read_csv(subm_file_path, dtype={'norm': object}, sep='\t')
		else:
			submission_df = pd.read_csv(os.path.join(subm_dir,subm_file_path), dtype={'norm': object}, sep='\t')
		
		if submission_df.shape[0] > 0:
			submission_merged_df = submission_df.merge(segment_df, on = ["file_id","segment_id"])
			submission_merged_df = submission_merged_df.drop('segment_id', axis=1)

			submission_df_sorted = submission_merged_df.sort_values(by=['start','end'])
			submission_dfs = pd.concat([submission_dfs, submission_df_sorted])

	submission_dfs.drop_duplicates(inplace = True)
	submission_dfs = submission_dfs.reset_index(drop=True)

	new_submission_dfs = change_class_type(submission_dfs, convert_task_column(task))

	return new_submission_dfs

def fill_epsilon_submission(sys, epsilon):
	"""
	replace the start of next segment with the end of last segment when difference is smaller than a threshold
	"""
	for i in range(sys.shape[0]-1):
		if sys.iloc[i]["end"] != sys.iloc[i+1]["start"] and abs(sys.iloc[i]["end"] - sys.iloc[i+1]["start"]) < epsilon:
			sys.iloc[i+1, sys.columns.get_loc('start')] = sys.iloc[i]["end"]
	
	return sys

def convert_task_column(task):
	"""
	Convert filename into column name
	"""
	if task == "norms" or task == "emotions":
		column_name = task.replace("s","")
	elif task == "changepoint":
		column_name = "timestamp"
	else:
		column_name = task

	return column_name

def add_type_column(ref_dir, hyp):
	"""
	Extract type column from ref and add it to hyp
	"""
	file_info = pd.read_csv(os.path.join(ref_dir,"docs","file_info.tab"), sep = "\t")
	file_info = file_info[["file_uid","type"]]
	file_info_uniq = file_info.drop_duplicates()
	hyp_type = hyp.merge(file_info_uniq, left_on = "file_id", right_on = "file_uid")

	return hyp_type

def filter_hyp_use_scoring_index(hyp, scoring_index):

	hyp_pruned = hyp.merge(scoring_index)

	return hyp_pruned

def merge_sys_time_periods(result_dict, llr_value, allowed_gap, merge_label, task):

	#print("merge input")
	#print(pprint.pprint(result_dict, width=200))
	result_array = []
	for key in result_dict.keys():
		time_array = sorted(result_dict[key], key=lambda d: d['start'])
		#print(f"Data for key {key}")
		#print(pprint.pprint(time_array, width=200))
		# Merge time array
		merged_time_array = []
		i = 0
		while i < len(time_array):
			first_time_period = time_array[i]
			current_time_period = time_array[i]
			hyp_uid_merge = time_array[i]["hyp_uid"]
			hyp_isTruncated = time_array[i]["hyp_isTruncated"]
			llr_merge = time_array[i]["llr"]
			llr_list = []
			llr_list.append(time_array[i]["llr"])
			if (task == "norms"):
				status_dict = {}
				status_dict[first_time_period['status']] = 1
			while ((i + 1 < len(time_array)) and
                               (float(current_time_period['end']) + allowed_gap > float(time_array[i + 1]['start'])) and
                               ((task != "norms") or
                                ((task == "norms") and
                                 ((merge_label == "class") or (first_time_period['status'] == time_array[i+1]['status']))))):
				i = i + 1
				llr_list.append(time_array[i]["llr"])
				current_time_period = time_array[i]
				if llr_value == "min_llr":
					llr_merge = min(llr_list)
				if llr_value == "max_llr":
					llr_merge = max(llr_list)
				if (task == "norms"):
					status_dict[current_time_period['status']] = 1
				hyp_isTruncated = True if hyp_isTruncated else time_array[i]["hyp_isTruncated"]
			med = {'start': first_time_period['start'], 'end': current_time_period['end'], 'llr': llr_merge, 'type': first_time_period['type'], 'hyp_uid': hyp_uid_merge, 'hyp_isTruncated': hyp_isTruncated}
			if (task == "norms"):
				st =list(status_dict.keys())
				st.sort()
				med['status'] = ",".join(st)
			merged_time_array.append(med)
			i = i + 1
		for item in merged_time_array:
			result_array.append({'content': item, 'group': key})

	#print("merge output")
	#print(pprint.pprint(result_array, width=200))
	#exit(0)
	return result_array

def indices(lst, item):

	return [i for i, x in enumerate(lst) if x == item]

def define_sys_boundary_general(sub_hyp, sub_ref_segments, list_of_lists, file_id, type):

	start = list(np.around(np.array(list(sub_ref_segments["start"])),3))
	end = list(np.around(np.array(list(sub_ref_segments["end"])),3))
	
	temp_label = {}
	temp_llr = {}
	final_label = {}
	final_llr = {}
	for i in range(len(sub_ref_segments)):
		temp_label[i] = []
		temp_llr[i] = []
		final_label[i] = []
		final_llr[i] = []

	count = 0
	for i,j in zip(start,end):
		for k in range(sub_hyp.shape[0]):
			sys_start = sub_hyp.iloc[k]["start"]
			sys_end = sub_hyp.iloc[k]["end"]

			tt1 = np.maximum(i, sys_start)
			tt2 = np.minimum(j, sys_end)    
			inter = (tt2 - tt1).clip(0)
			if inter > 0:
				temp_label[count].append(sub_hyp.iloc[k]["Class"])
				temp_llr[count].append(float(sub_hyp.iloc[k]["llr"]))
		count = count + 1

	count = 0
	for i in range(len(temp_llr)):
		for j in set(temp_label[i]):
			occur_index = indices(temp_label[i], j)
			llr_list = list(np.take(temp_llr[i], occur_index))
			max_llr = max(llr_list)

			final_label[count].append(j)
			final_llr[count].append(float(max_llr))
		count = count + 1

	# for i,j in zip(start,end):
	# 	print(i,j)
	for i in range(len(final_label)):
		for j in range(len(final_label[i])):
			list_of_lists.append([file_id,final_label[i][j],start[i],end[i],final_llr[i][j],file_id,type])

	return list_of_lists

def define_sys_boundary_text(sub_hyp, sub_ref_segments, list_of_lists, file_id, type):

	start = list(np.around(np.array(list(sub_ref_segments["start"])),3))
	end = list(np.around(np.array(list(sub_ref_segments["end"])),3))
	
	temp_label = {}
	temp_llr = {}
	final_label = {}
	final_llr = {}
	for i in range(len(sub_ref_segments)):
		temp_label[i] = []
		temp_llr[i] = []
		final_label[i] = []
		final_llr[i] = []

	count = 0
	for i,j in zip(start,end):
		for k in range(sub_hyp.shape[0]):
			sys_start = sub_hyp.iloc[k]["start"]
			sys_end = sub_hyp.iloc[k]["end"]

			tt1 = np.maximum(i, sys_start)
			tt2 = np.minimum(j, sys_end)    
			inter = (tt2 - tt1).clip(0)
			if inter > 0 or tt2 == tt1:
				temp_label[count].append(sub_hyp.iloc[k]["Class"])
				temp_llr[count].append(float(sub_hyp.iloc[k]["llr"]))
		count = count + 1

	count = 0
	for i in range(len(temp_llr)):
		for j in set(temp_label[i]):
			occur_index = indices(temp_label[i], j)
			llr_list = list(np.take(temp_llr[i], occur_index))
			max_llr = max(llr_list)

			final_label[count].append(j)
			final_llr[count].append(float(max_llr))
		count = count + 1

	# for i,j in zip(start,end):
	# 	print(i,j)
	for i in range(len(final_label)):
		for j in range(len(final_label[i])):
			list_of_lists.append([file_id,final_label[i][j],start[i],end[i],final_llr[i][j],file_id,type])

	return list_of_lists

def file_based_merge_sys(hyp, ref_segment):

	file_ids = get_unique_items_in_array(ref_segment['file_id'])
	list_of_lists = []
	for file_id in file_ids:
		sub_hyp = hyp.loc[hyp["file_id"] == file_id]
		type = list(sub_hyp["type"])[0]
		sub_ref_segments = ref_segment.loc[ref_segment["file_id"] == file_id]

		if type == "text":
			list_of_lists = define_sys_boundary_text(sub_hyp, sub_ref_segments, list_of_lists, file_id, type)
		else:
			list_of_lists = define_sys_boundary_general(sub_hyp, sub_ref_segments, list_of_lists, file_id, type)

	new_sys = pd.DataFrame(list_of_lists, columns=['file_id', 'Class', 'start', 'end', 'llr', 'file_uid', 'type'])
	new_sys["hyp_uid"] = ["H{}".format(x) for x in new_sys.index]
	new_sys["hyp_isTruncated"] = False
	return new_sys

def get_result_dict(sorted_df, merge_label, task):
        
	#print(f"get result_dict {merge_label}")
	result_dict = {}
	# if merge_label == "class":
	# 	for i in sorted_df.Class.unique():
	# 		result_dict_list = []
	# 		sub_df = sorted_df.loc[sorted_df["Class"] == i].reset_index()
	# 		for j in range(sub_df.shape[0]):
	# 			result_dict_list.append({"start": sub_df.iloc[j]['start'], "end": sub_df.iloc[j]['end'], "llr": sub_df.iloc[j]['llr'], "type": sub_df.iloc[j]['type']})
	# 		result_dict[i] = result_dict_list
	# if merge_label == "class-status":
	# 	for i in list(sorted_df.groupby(["Class","status"]).groups.keys()):
	# 		result_dict_list = []
	# 		sub_df = sorted_df.loc[(sorted_df["Class"] == i[0]) & (sorted_df["status"] == i[1])].reset_index()
	# 		for j in range(sub_df.shape[0]):
	# 			result_dict_list.append({"start": sub_df.iloc[j]['start'], "end": sub_df.iloc[j]['end'], "llr": sub_df.iloc[j]['llr'], "type": sub_df.iloc[j]['type']})
	# 		result_dict[i] = result_dict_list
	for i in sorted_df.Class.unique():
		result_dict_list = []
		sub_df = sorted_df.loc[sorted_df["Class"] == i].reset_index()
		hid = ""
		if (task == "norms"):
			for j in range(sub_df.shape[0]):
				result_dict_list.append({"start": sub_df.iloc[j]['start'], "end": sub_df.iloc[j]['end'], "llr": sub_df.iloc[j]['llr'], "type": sub_df.iloc[j]['type'], 'hyp_uid': sub_df.iloc[j]['hyp_uid'], "hyp_isTruncated": sub_df.iloc[j]['hyp_isTruncated'], "status":  sub_df.iloc[j]['status']})
		else:
			for j in range(sub_df.shape[0]):
				result_dict_list.append({"start": sub_df.iloc[j]['start'], "end": sub_df.iloc[j]['end'], "llr": sub_df.iloc[j]['llr'], "type": sub_df.iloc[j]['type'], 'hyp_uid': sub_df.iloc[j]['hyp_uid'], "hyp_isTruncated": sub_df.iloc[j]['hyp_isTruncated']})
		result_dict[i] = result_dict_list

	return result_dict

def get_merged_dict(file_ids, data_frame, text_gap, time_gap, llr_value, merge_label, task):

	#print(f"----- Get merge dict {text_gap}, {time_gap}, {llr_value}, {merge_label}, {task}")
	data_frame.set_index("file_id")
	final_df = pd.DataFrame()

	for file_id in file_ids:
		sorted_df = extract_df(data_frame, file_id)
                
		# Check file type to determine the gap of merging
		if list(sorted_df["type"])[0] == "text":
			if text_gap is not None:
				result_dict = get_result_dict(sorted_df, merge_label, task)
				result_array = merge_sys_time_periods(result_dict, llr_value, text_gap, merge_label, task)
				merged_df = convert_merge_dict_df(file_id, result_array, merge_label, task)
			else:
				merged_df = sorted_df

		if list(sorted_df["type"])[0] != "text":
			if time_gap is not None:
				result_dict = get_result_dict(sorted_df, merge_label, task)
				result_array = merge_sys_time_periods(result_dict, llr_value, time_gap, merge_label, task)
				merged_df = convert_merge_dict_df(file_id, result_array, merge_label, task)
			else:
				merged_df = sorted_df
		#print(f"alignfileid   {file_id}")
		#print(f"   {result_array}")
		#print(f"   {merged_df}")
		#exit(0)
		merged_sorted_df = merged_df.sort_values(by=['Class','start','end'])
		final_df = pd.concat([final_df,merged_sorted_df], ignore_index = True) 

	return final_df

def convert_merge_dict_df(file_id, results_array, merge_label, task):
	"""
	Convert dictionary of norm/emotion into a dataframe
	"""

	file_ids = []
	file_uids = []
	starts = []
	ends = []
	Class = []
	status = []
	llrs = []
	types = []
	hypuids = []
	hypistrunc = []
        	
	#print(f"Converting {task}")
	for segment in results_array:
		file_ids.append(file_id)
		file_uids.append(file_id)
		starts.append(float(segment['content']['start']))
		ends.append(float(segment['content']['end']))
		Class.append(segment['group'])
		if (task == "norms"):
			status.append(segment['content']['status'])
		llrs.append(segment['content']['llr'])
		types.append(segment['content']['type'])
		hypuids.append(segment['content']['hyp_uid'])
		hypistrunc.append(segment['content']['hyp_isTruncated'])
                
	if (task == "norms"):
		result_df = pd.DataFrame({"file_id":file_ids,"file_uid":file_uids,"Class":Class,"start":starts,"end":ends,"hyp_uid":hypuids, "hyp_isTruncated": hypistrunc,"status":status,"llr":llrs,"type":types})
	else:
		result_df = pd.DataFrame({"file_id":file_ids,"file_uid":file_uids,"Class":Class,"start":starts,"end":ends,"hyp_uid":hypuids, "hyp_isTruncated": hypistrunc, "llr":llrs,"type":types})
                
	return result_df

def preprocess_submission_file(subm_dir, ref_dir, scoring_index, task, submission_format = None, gap_allowed = False):

	if submission_format == "open":
		hyp = concatenate_openccu_submission_file(subm_dir, ref_dir, task)
	else:
		hyp = concatenate_submission_file(subm_dir, task)

	if hyp.shape[0] > 0:
		hyp_type = add_type_column(ref_dir, hyp)
		hyp_final = filter_hyp_use_scoring_index(hyp_type, scoring_index)
	else:
		hyp_final = hyp

	if task in ["valence_continuous","arousal_continuous"] and gap_allowed:
		hyp_final = extend_gap_segment(hyp_final, "hyp")
		hyp_final = add_start_end_sys(hyp_final, ref_dir)

	if hyp_final.shape[0] > 0:
		hyp_final['hyp_uid'] = [ "H"+str(s) for s in range(len(hyp_final['file_id'])) ] ### This is a unique HYP ID
		hyp_final['hyp_isTruncated'] = False

	return hyp_final

def add_start_end_sys(submission_df, ref_dir):

	file_info_df = pd.read_csv(os.path.join(ref_dir,"docs","file_info.tab"), sep = "\t")
	file_ids = get_unique_items_in_array(submission_df['file_id'])

	for i in file_ids:
		df = extract_df(submission_df, i)
		type = list(df["type"])[0]
		length = file_info_df["length"][file_info_df['file_uid'] == i].values[0]
		start = min(list(df["start"]))
		end = max(list(df["end"]))

		if type in ["audio","video"]:
			if start > 0:
				df_add = pd.DataFrame({"file_id": i, "start": 0, "end": start, "Class": silence_string, "file_uid": i, "type": type}, index=[0])
				submission_df = pd.concat([submission_df, df_add], ignore_index=True)

			if end < length:
				df_add = pd.DataFrame({"file_id": i, "start": end, "end": length, "Class": silence_string, "file_uid": i, "type": type}, index=[0])
				submission_df = pd.concat([submission_df, df_add], ignore_index=True)
			
		if type == "text":
			if start > 0:
				df_add = pd.DataFrame({"file_id": i, "start": 0, "end": start - 1, "Class": silence_string, "file_uid": i, "type": type}, index=[0])
				submission_df = pd.concat([submission_df, df_add], ignore_index=True)

			if end < length:
				df_add = pd.DataFrame({"file_id": i, "start": end + 1, "end": length, "Class": silence_string, "file_uid": i, "type": type}, index=[0])
				submission_df = pd.concat([submission_df, df_add], ignore_index=True)			

	submission_df = submission_df.sort_values(by=["file_id","start","end"]).reset_index(drop=True)
	return submission_df

def merge_sys_instance(hyp, text_gap, time_gap, llr_value, merge_label, task):
        
	#print(f"Merging hyp {text_gap} {time_gap} {llr_value} {merge_label} {task}")
	if text_gap is None and time_gap is None:
		return hyp
	if (len(hyp) == 0):  ### make an enpty structure
		return hyp

	# Split input_file into parts based on file_id column
	file_ids = get_unique_items_in_array(hyp['file_id'])
	# Generate file_id map for vote processing
	merged_df = get_merged_dict(file_ids, hyp, text_gap, time_gap, llr_value, merge_label, task)

	return merged_df

def extend_gap_segment(df, data):
	"""
	Extend the gap between end of previous segment and start of next segment in reference.

	e.g.
	From
	Seg  Start  End
	Seg1 45.2   60.2
	Seg2 60.201 75.201
	To result
	Seg  Start  End
	Seg1 45.2   60.201
	Seg2 60.201   75.201

	Parameters
	----------
	ref

	Returns
	-------
	ref_sorted: data frame after extending the gap

	"""

	gap_map = {"text": 10, "audio": 1, "video": 1}
	if data == "ref":
		class_type = list(df["Class_type"])[0]
	for i in range(1,df.shape[0]):
		if df.iloc[i]["file_id"] == df.iloc[i-1]["file_id"]:
			diff = round(df.iloc[i]["start"] - df.iloc[i-1]["end"],3)
			gap = gap_map[df.iloc[i]["type"]]

			if df.iloc[i]["type"] == "text":
				if diff < gap:
					df.iloc[i-1, df.columns.get_loc('end')] = df.iloc[i]["start"] - 1
				else:
					if data == "ref":
						df.loc[len(df.index)] = [df.iloc[i]["file_id"], silence_string, df.iloc[i-1, df.columns.get_loc('end')] + 1, df.iloc[i]["start"] - 1, class_type, df.iloc[i]["type"], df.iloc[i]["length"]]
					if data == "hyp":
						df.loc[len(df.index)] = [df.iloc[i]["file_id"], df.iloc[i-1, df.columns.get_loc('end')] + 1, df.iloc[i]["start"] - 1, silence_string, df.iloc[i]["file_id"], df.iloc[i]["type"]]
			elif df.iloc[i]["type"] in ["audio", "video"]:
				if diff < gap:
					df.iloc[i-1, df.columns.get_loc('end')] = df.iloc[i]["start"]
				else:
					if data == "ref":
						df.loc[len(df.index)] = [df.iloc[i]["file_id"], silence_string, df.iloc[i-1, df.columns.get_loc('end')], df.iloc[i]["start"], class_type, df.iloc[i]["type"], df.iloc[i]["length"]]
					if data == "hyp":
						df.loc[len(df.index)] = [df.iloc[i]["file_id"], df.iloc[i-1, df.columns.get_loc('end')], df.iloc[i]["start"], silence_string, df.iloc[i]["file_id"], df.iloc[i]["type"]]

	df_sorted = df.sort_values(by=["file_id","start","end"]).reset_index(drop=True)
	return df_sorted

def extract_df(df, file_id):
	"""
	Extract sub ref/hyp for specific file_id
	"""
	partial_df = df[df['file_id']==file_id]
	sorted_df = partial_df.sort_values(by=['start','end'])

	return sorted_df

def change_class_type(df, class_type):
	"""
	Change difference column names into a general column name 
	"""
	new_df = df.rename(columns = {class_type: "Class"})

	return new_df

def formatNumber(num):
	"""
	Convert integer number into integer format
	"""
	if num % 1 == 0:
		return str(int(num))
	else:
		return str(num)

def replace_hyp_norm_mapping(sub_mapping_df, hyp, act):
	"""
	Merge mapping file with hyp for specific norm
	"""
	sys_norm_list = list(sub_mapping_df.sys_norm)
	sub_hyp = hyp[hyp.Class.isin(sys_norm_list)]
	new_sub_hyp = sub_mapping_df.merge(sub_hyp, left_on='sys_norm', right_on='Class')
	new_sub_hyp = new_sub_hyp[["file_id","ref_norm","start","end","status","llr","type","hyp_uid","file_uid","hyp_isTruncated"]]
	new_sub_hyp.rename(columns={"ref_norm": "Class"}, inplace=True)
	new_sub_hyp.drop_duplicates(inplace = True)
	final_sub_hyp = pd.concat([new_sub_hyp, hyp.loc[(hyp.Class == act)]])
        
	return final_sub_hyp

def generate_alignment_statistics(ali, task, output_dir, info_dict = None):
        """
        Generate statistics from the alignment file.
        """
        #print(ali)
        #print(task)
        if (task == 'norm'):
                count_matrix = {}
                for index, row in ali.iterrows():
                        rstr = row['ref_status']
                        hstr = row['hyp_status']
                        for Class in ['all', row['class']]:
                                if (Class not in count_matrix):
                                        count_matrix[Class] = {}
                                if (rstr not in count_matrix[Class]):
                                        count_matrix[Class][rstr] = {}
                                if (hstr not in count_matrix[Class][rstr]):
                                        count_matrix[Class][rstr][hstr] = 0
                                count_matrix[Class][rstr][hstr] += 1
                        
                # print("\nInstance Count Matrix: Rows == Reference")
                # print(mat.columns.tolist())
                # print(mat.index.tolist())
                # print(pd.wide_to_long(pd.DataFrame(count_matrix), '', mat.columns.tolist(), mat.index.tolist()))
                file1 = open(os.path.join(output_dir, "instance_alignment_status_confusion.tab"), 'w')
                file1.write("class	ref_status	hyp_status	metric	value\n")
                for Class in count_matrix.keys():
                        #print(f"Class = {Class}")
                        mat = pd.DataFrame(count_matrix[Class]).transpose()
                        mat = mat.sort_index()
                        mat = mat.reindex(sorted(mat.columns), axis=1)
                        #print(mat)
                        for rstr in count_matrix[Class].keys():
                            for hstr in count_matrix[Class][rstr].keys():
                                file1.write(f"{Class}	{rstr}	{hstr}	number_instance	{count_matrix[Class][rstr][hstr]}\n")
                file1.close()

        at_MinLLR = {'all': { 'cd': 0, 'fa': 0, 'md': 0} }
        for index, row in ali.iterrows():
                if (row['class'] not in at_MinLLR):
                        at_MinLLR[row['class']] = { 'cd': 0, 'fa': 0, 'md': 0 } 
                if (row['eval'] == "mapped"):
                        at_MinLLR['all']['cd'] += 1
                        at_MinLLR[row['class']]['cd'] += 1
                else:
                        if (row['ref'] == '{}'):
                                at_MinLLR['all']['fa'] += 1
                                at_MinLLR[row['class']]['fa'] += 1
                        else:
                                at_MinLLR['all']['md'] += 1
                                at_MinLLR[row['class']]['md'] += 1

        #print("\nat MinMLLR")
        #print(pd.DataFrame(at_MinLLR))
        file1 = open(os.path.join(output_dir, "instance_alignment_class_stats.tab"), 'w')
        file1.write("class	metric	value\n")
        for Class in at_MinLLR.keys():
                for met in at_MinLLR[Class].keys():
                        file1.write(f"{Class}	number_{met}	{at_MinLLR[Class][met]}\n")
        file1.close()

        def get_val(str, attrib):
                match = re.match('.*[{,]('+attrib+')=([^,}]+)', str)
                if (match is not None):
                        return(match.group(2))
                return(None)

        #get_val("{iou=0.001,intersection=1.000,union=825.000}","iou")
        #get_val("{iou=0.001,intersection=1.000,union=825.000}","intersection")
        #get_val("{iou=0.001,intersection=1.000,union=825.000}","union")

        all_llrs = sorted(ali[ali['sys'] != '{}']['llr'])
        fig, ax = plt.subplots(1,
                               2 if task in ['cd'] else 4,
                               figsize = (6 if task in ['cd'] else 12, 4))
        ax_id = 0;
        params = []
        if (task in ['norm', 'emotion']):
                params = ['iou', 'intersection']
        if (task in ['cd']):
                params = ['delta_cp']
        for param in params:
                data = [ x for x in [ get_val(x,param) for x in ali[ali['eval'] == 'mapped']['parameters'] ] if x is not None ]
                if (len(data) >= 2):
                        data = sorted([float(x) for x in data])  ## sort and float
                        maxx = np.maximum(1, data[-1])
                        ax[ax_id].hist(data, bins = np.linspace(0, maxx, num=100, endpoint = True))
                        ax[ax_id].set_title("Histogram (Mean={:.3f})".format(np.mean(data)))
                else:
                        ax[ax_id].set_title("NO SYSTEM OUTPUT")               
                ax[ax_id].set_xlabel(param)
                ax_id += 1
                
        if (len(all_llrs) >= 2):
                bins = np.linspace(all_llrs[0], all_llrs[-1], num=100, endpoint = True)
                ax[ax_id].hist(ali[ (ali['sys'] != '{}') & (ali['eval'] == 'mapped') ]['llr'], bins = bins, histtype=u'step', color="#00FF00", label="Target")
                ax[ax_id].hist(ali[ (ali['sys'] != '{}') & (ali['eval'] != 'mapped') ]['llr'], bins = bins, histtype=u'step', color="#FF0000", label="NonTarget")
                ax[ax_id].legend(loc='upper right')
                ax[ax_id].set_title("Histogram")
        else:
                ax[ax_id].set_title("NO SYSTEM OUTPUT")
        ax[ax_id].set_xlabel("LLR")

        # Show plot
        out = os.path.join(output_dir, "instance_alignment_graphs.png")
        fig.savefig(out)
        if (info_dict is not None):
                info_dict.append({ 'task': task, 'graph_type': 'instance_alignment', 'graph_factor': 'overall', 'graph_factor_value': 'all', 'correctness_constraint': "n/a", 'filename': out})
        plt.close()
        
def generate_alignment_file(ref, hyp, task):
	"""
	Generate alignment file using ihyp and ref
	"""
	def categorise(row):
		if row['tp'] == 1 and row['fp'] == 0:
			return 'mapped'
		return 'unmapped'
	def categorise_score(row):
		if row['tp'] == 1 and row['fp'] == 0:
			return 'CD'
		if row['tp'] == 0 and row['fp'] == 1:
			return 'FA'
		if row['forgive-fp'] == 1:
			return 'F-FA'
		return 'MD'

	hyp["eval"] = hyp.apply(lambda row: categorise(row), axis=1)
	hyp["eval_score"] = hyp.apply(lambda row: categorise_score(row), axis=1)

	#print(f">> Generate alignment file for task {task}")
	#print("REF")
	#print(ref)
	#print("HYP")
	#print(hyp)
	#exit(0)
	if task in ["norm","emotion"]:
                ### Refs are in the hyp df
		# ref_format = ref.copy()
		# ref_format['start'] = [formatNumber(x) for x in ref['start']]
		# ref_format['end'] = [formatNumber(x) for x in ref['end']]
		# ref_format['sort'] = [x for x in ref['start']]
		#print("ref_format")
		#print(ref_format)
		
		hyp_format = hyp.copy()
		#print(hyp_format)
		### Duplicate the rows for the MD from the correctness constraint
		new_md = hyp_format[(hyp_format.fp == 1) & (hyp_format.md == 1)].copy()
		new_md[['fp', 'md', 'eval_score']] = [0, 1, 'MD']
		if (task == 'norm'):
			new_md[['hyp_status']] = ['EMPTY_NA']
		hyp_format = pd.concat([hyp_format, new_md])
		#print(hyp_format)
		#exit(0)
                
		hyp_format['sort'] = [r if (math.isnan(h) ) else h for h, r in zip(hyp_format['start_hyp'], hyp_format['start_ref']) ]
		hyp_format['start_ref'] = [formatNumber(x) for x in hyp_format['start_ref']]
		hyp_format['end_ref'] = [formatNumber(x) for x in hyp_format['end_ref']]
		hyp_format['start_hyp'] = [formatNumber(x) for x in hyp_format['start_hyp']]
		hyp_format['end_hyp'] = [formatNumber(x) for x in hyp_format['end_hyp']]

		hyp_format["ref"] = "{start=" + hyp_format["start_ref"].astype(str) + ",end=" + hyp_format["end_ref"].astype(str) + "}"
		hyp_format["sys"] = "{start=" + hyp_format["start_hyp"].astype(str) + ",end=" + hyp_format["end_hyp"].astype(str) + "}"

		hyp_format['IoU_f'] =               hyp_format['IoU'].apply(              lambda x: 'iou={:.3f}'.format(x))
		hyp_format['intersection_f'] =      hyp_format['intersection'].apply(     lambda x: 'intersection={:.3f}'.format(x))
		hyp_format['union_f'] =             hyp_format['union'].apply(            lambda x: 'union={:.3f}'.format(x))
		hyp_format['shifted_sys_start_f'] = hyp_format['shifted_sys_start'].apply(lambda x: 'shifted_start_hyp={:.3f}'.format(x))
		hyp_format['shifted_sys_end_f'] =   hyp_format['shifted_sys_end'].apply(  lambda x: 'shifted_end_hyp={:.3f}'.format(x))
		hyp_format['pct_tp_f'] =            hyp_format['pct_tp'].apply(           lambda x: 'pct_temp_tp={:.3f}'.format(x))
		hyp_format['pct_fp_f'] =            hyp_format['pct_fp'].apply(           lambda x: 'pct_temp_fp={:.3f}'.format(x))
		hyp_format['collar'] =              hyp_format['scale_collar'].apply(     lambda x: 'collar={:.3f}'.format(x))
		hyp_format['type'] =                hyp_format['type'].apply(             lambda x: 'type={}'.format(x))
		hyp_format['eval_str'] =            hyp_format['eval_score'].apply(       lambda x: 'eval={}'.format(x))

		hyp_format['parameters'] = '{' + hyp_format['IoU_f'] + ',' + hyp_format['intersection_f'] + ',' + hyp_format['union_f'] + ',' + hyp_format['shifted_sys_start_f'] + ',' + hyp_format['shifted_sys_end_f'] + ',' + hyp_format['pct_tp_f'] + ',' + hyp_format['pct_fp_f'] + ',' + hyp_format['collar'] + ',' + hyp_format['type'] + ',' + hyp_format['eval_str'] + '}'
#		hyp_format['parameters'] = '{' + hyp_format['IoU_f'] + ',' + hyp_format['intersection_f'] + ',' + hyp_format['union_f'] + ',' + hyp_format['shifted_sys_start_f'] + ',' + hyp_format['shifted_sys_end_f'] + ',' + hyp_format['pct_tp_f'] + ',' + hyp_format['pct_fp_f'] + ',' + hyp_format['collar'] + ',' + hyp_format['type'] + '}'

		hyp_format.loc[hyp_format.eval_score == 'FA', "ref"] = "{}"  ### start_ref is now a string!!!
		hyp_format.loc[hyp_format.eval_score == 'MD', "sys"] = "{}"  ### start_ref is now a string!!!
		#hyp_format.loc[hyp_format["eval"] == "unmapped", "ref"] = "{}"
		#hyp_format.loc[hyp_format.eval_score == 'FA', "parameters"] = hyp_format[hyp_format.eval_score == 'FA','eval_str'].apply(lambda x: "{" + "pct_temp_tp=0.0,pct_temp_fp=1.0,{}".format(x) + "}" )
		#hyp_format.loc[hyp_format.eval_score == 'MD', "parameters"] = hyp_format[hyp_format.eval_score == 'MD','eval_str'].apply(lambda x: "{" + "pct_temp_tp=0.0,pct_temp_fp=0.0,{}".format(x) + "}")
		hyp_format.loc[hyp_format.eval_score == 'FA', "parameters"] = "{pct_temp_tp=0.0,pct_temp_fp=1.0,eval=FA}"
		hyp_format.loc[hyp_format.eval_score == 'MD', "parameters"] = "{pct_temp_tp=0.0,pct_temp_fp=0.0,eval=MD}"

		#print("hyp_format befor filtering columns")
		#print(hyp_format)
		#exit(0)
                
                ### Filter columns
		if (task == "norm"):
			hyp_format.loc[hyp_format["eval_score"] == "FA", "status"] = "EMPTY_NA"  ### This is the REF - to be renamed below
			hyp_format = hyp_format[["Class","file_id","eval","ref","sys","llr","parameters","sort","status", "hyp_status"]]
			hyp_format = hyp_format.rename(columns={"status": "ref_status"})
		else:
			hyp_format = hyp_format[["Class","file_id","eval","ref","sys","llr","parameters","sort"]]                        


                ### REF as all in the hyp structure
                # #### These are the unmapped refs
		# ref_new = ref_format.copy()
		# ref_new["ref"] = "{start=" + ref_format["start"].astype(str) + ",end=" + ref_format["end"].astype(str) + "}"
		# if (task == "norm"):
		#         ref_new = ref_new[["file_id","Class","ref","sort","status"]]
		#         ref_new = ref_new.rename(columns={"status": "ref_status"})
		#         ref_new['hyp_status'] = "EMPTY_NA"
		# else:
		#         ref_new = ref_new[["file_id","Class","ref","sort"]]
		# ref_new = ref_new.loc[~(ref_new["ref"].isin(hyp_format["ref"]))]
		# ref_new["eval"] = "unmapped"
		# ref_new["sys"] = "{}"
		# ref_new["parameters"] = "{}"
		# ref_new["llr"] = np.nan
		# print(ref_new)

		# alignment = pd.concat([hyp_format, ref_new])

		alignment = hyp_format.rename(columns={'Class':'class'})
		#print("final alignment")
		#print(hyp_format)
		#exit(0)
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
	"""
	Generate alignment file for no match using ref
	"""
	if task in ["norm","emotion"]:

		ref_format = ref.copy()
		#print(ref_format)
		ref_format['start'] = [formatNumber(x) for x in ref['start']]
		ref_format['end'] = [formatNumber(x) for x in ref['end']]
		ref_format['sort'] = [x for x in ref['start']]

		ref_new = ref_format.loc[ref_format.Class.str.contains('NO_SCORE_REGION')==False].copy()
		ref_new["ref"] = "{start=" + ref_format["start"].astype(str) + ",end=" + ref_format["end"].astype(str) + "}"
		#ref_new = ref_new[["file_id","Class","ref"]]
		ref_new["eval"] = "unmapped"
		ref_new["sys"] = "{}"
		ref_new["parameters"] = "{}"
		ref_new["llr"] = np.nan
		ref_new["sort"] = ref_format['sort']
		ref_new = ref_new.rename(columns={'Class':'class'})

		co = ["class","file_id","eval","ref","sys","llr","parameters","sort"]
		if (task == "norm"):
			ref_new['hyp_status'] = "EMPTY_NA"
			ref_new = ref_new.rename(columns={'status': "ref_status"})
			co.append("ref_status")
			co.append("hyp_status")
		ref_new = ref_new[ co ];
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

def generate_scoring_parameter_file(args):

	metrics = []
	values = []

	for i in vars(args):
		if i != 'func':
			metrics.append(i)
			Value = vars(args)[i]
			if (fnmatch.fnmatch(i, "*_dir") or fnmatch.fnmatch(i, "*_file")) and Value is not None:
				values.append(os.path.abspath(Value))
			else:
				values.append(Value)

	metrics.append("git.commit")

	try:
		lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
		commit_id = subprocess.check_output(["git", "--git-dir="+ lib_path +".git", "show", "--oneline", "-s", "--no-abbrev-commit","--pretty=format:%H--%aI"]).strip()
	except:
		commit_id = "None"

	values.append(commit_id)

	sp_d = {}
	sp_d["metric"] = metrics
	sp_d["value"] = values

	sp_df = pd.DataFrame(data = sp_d)
	sp_df = sp_df.replace(np.nan, "None")

	sp_df.to_csv(os.path.join(args.output_dir, "scoring_parameters.tab"), sep = "\t", index = None)

def make_pr_curve_for_cd(apScore, task, title = "", output_dir = ".", info_dict = None):
    """ Plot a Precision Recall Curve for the data.
    
    Parameters
    ----------
    apScore:
        - { IoU: [**ap**, **precision** (1darray), **recall** (1darray) , .... }
    info_dict:
        - an array of dictionaries describing the graphs

    Returns
    -------
    """

    #print("Making Precision-Recall Curves by Genre")
    for iou, class_data in apScore.items():
        iou_str = str(iou).replace('=', '_')
        for genre in set(class_data['type']):
            out = os.path.join(output_dir, f"pr_{iou_str}_type_{genre}.png")
            fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
            ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.1),
                   ylim=(0, 1), yticks=np.arange(0, 1, 0.1))
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f"{title}, Correctness:{iou}, Type={genre}")
            dlist = []
            for index, row in class_data[class_data['type'] == genre].iterrows():
                ax.plot(row['recall'], row['precision'], linewidth=1.0, label=row['Class'])
                dlist.append(np.array([ row['recall'], row['precision'] ]))
            agg_recall, agg_precision, agg_stderr = aggregate_xy(dlist)
            ax.plot(agg_recall, agg_precision, linewidth=1.0, label="Average")
            #print("    Saving plot {}".format(out))
            if (info_dict is not None):
                    info_dict.append({ 'task': task, 'graph_type': 'pr_curve', 'graph_factor': 'genre', 'graph_factor_value': genre, 'correctness_constraint': iou, 'filename': out})
            plt.legend(loc='upper right')
            plt.savefig(out)
            plt.close()
    
    #print("Making Precision-Recall Curves by Class")
    ### Need to Re-order to be able to iterate over classes
    for iou, class_data in apScore.items():
        iou_str = str(iou).replace('=', '_')
        for Class in set(class_data['Class']):
            out = os.path.join(output_dir, f"pr_{iou_str}_class_{Class}.png")
            fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
            ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.1),
                   ylim=(0, 1), yticks=np.arange(0, 1, 0.1))
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f"{title} Correctness:{iou} Class={Class}")
            dlist = []
            has_all = False
            for index, row in class_data[class_data['Class'] == Class].iterrows():
                ax.plot(row['recall'], row['precision'], linewidth=1.0, label=row['type'])
                dlist.append(np.array([ row['recall'], row['precision'] ]))
                if (row['type'] == "all"):
                        has_all = True
            if (not has_all):                       
                    agg_recall, agg_precision, agg_stderr = aggregate_xy(dlist)
                    ax.plot(agg_recall, agg_precision, linewidth=1.0, label="Average")
            #print("    Saving plot {}".format(out))
            if (info_dict is not None):
                    info_dict.append({ 'task': task, 'graph_type': 'pr_curve', 'graph_factor': 'class', 'graph_factor_value': Class, 'correctness_constraint': iou, 'filename': out})           
            plt.legend(loc='upper right')
            plt.savefig(out)
            plt.close()


    return(info_dict)
	


def make_pr_curve(apScore, task, title = "", output_dir = ".", info_dict = None):
    """ Plot a Precision Recall Curve for the data.
    
    Parameters
    ----------
    apScore:
        - { IoU: [**ap**, **precision** (1darray), **recall** (1darray) , .... }
    info_dict:
        - an array of dictionaries describing the graphs

    Returns
    -------
    """
    #print("Making Precision-Recall Curves by Genre")
    for iou, class_data in apScore.items():
        iou_str = str(iou).replace(':', '_')        
        genres = list(set( [ x['type'] for x in class_data ] ) )
        for genre in genres:
            out = os.path.join(output_dir, f"pr_{iou_str}_type_{genre}.png")
            fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
            ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.1),
                   ylim=(0, 1), yticks=np.arange(0, 1, 0.1))
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f"{title}, Correctness:{iou}, Type={genre}")
            dlist = []
            for row in class_data:
                if (row['type'] == genre):
                    if (row['prcurve:recall'] is not None):
                            ax.plot(row['prcurve:recall'], row['prcurve:precision'], linewidth=1.0, label=row['Class'])
                            dlist.append(np.array([ row['prcurve:recall'], row['prcurve:precision'] ]))
                    else:
                            ax.plot([-1], [-1], linewidth=1.0, label=row['Class'] + "-No Sys Output")

            if (len(dlist) > 0):
                    agg_recall, agg_precision, agg_stderr = aggregate_xy(dlist)
                    ax.plot(agg_recall, agg_precision, linewidth=1.0, label="Average")
            else:
                    ax.plot([-1], [-1], linewidth=1.0, label="No Average")
            #print("    Saving plot {}".format(out))
            if (info_dict is not None):
                    info_dict.append({ 'task': task, 'graph_type': 'pr_curve', 'graph_factor': 'genre', 'graph_factor_value': genre, 'correctness_constraint': iou, 'filename': out})
            plt.legend(loc='upper right')
            plt.savefig(out)
            plt.close()
    
    #print("Making Precision-Recall Curves by Class")
    for iou, class_data in apScore.items():
        iou_str = str(iou).replace(':', '_')        
        classes = list(set( [ x['Class'] for x in class_data ] ) )
        for Class in classes:
            out = os.path.join(output_dir, f"pr_{iou_str}_class_{Class}.png")
            fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
            ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.1),
                   ylim=(0, 1), yticks=np.arange(0, 1, 0.1))
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f"{title}, Correctness:{iou}, Class={Class}")
            dlist = []
            for row in class_data:
                if (row['Class'] == Class):
                    if (row['prcurve:recall'] is not None):
                            ax.plot(row['prcurve:recall'], row['prcurve:precision'], linewidth=1.0, label=row['type'])
                            dlist.append(np.array([ row['prcurve:recall'], row['prcurve:precision'] ]))
                    else:
                            ax.plot([-1], [-1], linewidth=1.0, label=row['type'] + "-No Sys Output")
            if (len(dlist) > 0):
                    agg_recall, agg_precision, agg_stderr = aggregate_xy(dlist)
                    ax.plot(agg_recall, agg_precision, linewidth=1.0, label="Average")
            else:
                    ax.plot([-1], [-1], linewidth=1.0, label="No Average")
            #print("    Saving plot {}".format(out))
            if (info_dict is not None):
                    info_dict.append({ 'task': task, 'graph_type': 'pr_curve', 'graph_factor': 'class', 'graph_factor_value': Class, 'correctness_constraint': iou, 'filename': out})
            plt.legend(loc='upper right')
            plt.savefig(out)
            plt.close()

    return(info_dict)
