import os
import pandas as pd
from pathlib import Path
from .preprocess_reference import *
import logging

logger = logging.getLogger('VALIDATION')

def global_file_checks(task, reference_dir, submission_dir):

	ref = preprocess_reference_dir(ref_dir = reference_dir, task = task)
	index_file_path, subm_file_dict = check_index_get_submission_files(ref, submission_dir)
	check_submission_files(submission_dir, index_file_path, subm_file_dict)

	return ref, subm_file_dict

def individual_file_check(task, subm_file_path, column_map, header_map, processed_label, subm_file, length, ref_df, norm_list):
	
	if task == "norms":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_start_small_end(subm_file_path) and
			check_start_end_timestamp_within_length(subm_file_path, task, length))
	
	if task == "emotions":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_emotion_id(subm_file_path) and
			check_start_small_end(subm_file_path) and
			check_start_end_timestamp_within_length(subm_file_path, task, length))

	if task == "valence_continuous" or task == "arousal_continuous":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_start_small_end(subm_file_path) and
			check_time_no_gap(subm_file_path, header_map[task]) and
			check_duration_equal(subm_file_path, ref_df) and
			check_start_end_timestamp_within_length(subm_file_path, task, length) and
			check_value_range(subm_file_path, task))

	if task == "changepoint":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_start_end_timestamp_within_length(subm_file_path, task, length))

	if task == "index":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_data_type(subm_file_path, header_map[task]))

	if task == "ndmap":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_data_type(subm_file_path, header_map[task]) and
			check_ref_norm(norm_list, subm_file_path))
	
	return file_checks

def check_submission_files(subm_dir, index_file_path, subm_file_dict):

	subm_files = [path.as_posix() for path in Path(subm_dir).rglob('*.tab') if path.as_posix() != index_file_path]

	if set(subm_files) != set([i["path"] for i in subm_file_dict.values()]):
		logger.error('Invalid directory {}:'.format(subm_dir))

		# Check if we had not enough docs in the  reference files
		invalid_file = set(subm_files) - set([i["path"] for i in subm_file_dict.values()].values())
		if invalid_file:
			logger.error("Additional file(s) {} have been found in submission {}".format(invalid_file, subm_dir))
 
		# Check whether we had too many docs in the reference files
		invalid_file = set([i["path"] for i in subm_file_dict.values()].values()) - set(subm_files)
		if invalid_file:
			logger.error('The following document(s) {} were not found: {}'.format(invalid_file, subm_dir))
			
		logger.error('Validation failed')
		exit(1)
	else:
		pass

	return None

def check_file_exist(file, dir):

	if os.path.exists(file):
		pass
	else:
		logger.error('No file {} found in {}'.format(file, dir))
		logger.error('Validation failed')
		exit(1)

def check_valid_tab(file):

	try:
		pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
		return True
	except Exception as e:
		logger.error('{} is not a valid tab file'.format(file))
		return False

def check_column_number(file, columns_number):

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

	if df.shape[1] != columns_number:
		logger.error('Invalid file {}:'.format(file))
		logger.error('File {} should contain {} columns.'.format(file, columns_number))
		return False
	return True

def check_valid_header(file, header):

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

	if set(list(df.columns)) != set(header):
		logger.error('Invalid file {}:'.format(file))

		# Check if we had not enough docs in the  reference files
		invalid_header = set(header) - set(list(df.columns))
		if invalid_header:
			logger.error("Header of '{}' is missing following fields: {}".format(file, invalid_header))
 
		# Check whether we had too many docs in the reference files
		invalid_header = set(list(df.columns)) - set(header)
		if invalid_header:
			logger.error("Additional field(s) '{}'' have been found in header of {}".format(invalid_header, file))

		return False
	return True

def check_output_records(file, processed_label):

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	if processed_label == False and df.shape[0] != 0:
		logger.error("Output records have been found in submission file {} with False is_processed label".format(file))
		return False
	return True

def check_data_type(file, header_type):

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	if df.shape[0] != 0:
		res = df.dtypes
		invalid_type_column = []
		for i in df.columns:
			if res[i] != header_type[i]:
				invalid_type_column.append(i)

		if len(invalid_type_column) > 0:
			logger.error('Invalid file {}:'.format(file))
			logger.error("The data type of column {} in file {} is invalid".format(invalid_type_column, file))
			return False
	return True

def check_emotion_id(file):

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	if df.shape[0] != 0:
		invalid_emotion = []
		for i in df["emotion"]:
			if i not in ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]:
				invalid_emotion.append(i)

		if len(invalid_emotion) > 0:
			logger.error('Invalid file {}:'.format(file))
			logger.error("Additional emotion(s) '{}'' have been found in {}".format(set(invalid_emotion), file))
			return False
	return True

def check_start_small_end(file):

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	if df.shape[0] != 0:
		for i in range(df.shape[0]):
			if df.iloc[i]["start"] >= df.iloc[i]["end"]:
				logger.error('Invalid file {}:'.format(file))
				logger.error("Start is equal to /larger than end in {}".format(file))
				return False
	return True

def check_time_no_gap(file, header_type):

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	df_sorted = df.sort_values(by=['start','end'])

	if header_type["start"] == "int":
		for i in range(df_sorted.shape[0]-1):
			if df_sorted.iloc[i]["end"] + 1 != df_sorted.iloc[i+1]["start"]:
				logger.error('Invalid file {}:'.format(file))
				logger.error("There are some gaps in timestamp of {}".format(file))
				return False
		return True
	
	if header_type["start"] == "float":
		for i in range(df_sorted.shape[0]-1):
			if df_sorted.iloc[i]["end"] != df_sorted.iloc[i+1]["start"]:
				logger.error('Invalid file {}:'.format(file))
				logger.error("There are some gaps in timestamp of {}".format(file))
				return False
		return True

def check_duration_equal(file, ref_df):

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

	def calculate_duration_vd_ad(df):

		start = min(list(df["start"]))
		end = max(list(df["end"]))

		return start, end

	start_ref, end_ref = calculate_duration_vd_ad(ref_df)
	start_hyp, end_hyp = calculate_duration_vd_ad(df)
	
	if not ((start_ref == start_hyp) and (end_ref == end_hyp)):
		logger.error('Invalid file {}:'.format(file))
		logger.error("The duration of {} is different from the duration of reference".format(file))
		return False

	return True

def check_fileid_index_match(file, file_id):

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	if df.shape[0] != 0:
		invalid_fileid = []
		for i in df["file_id"]:
			if i != file_id:
				invalid_fileid.append(i)

		if len(invalid_fileid) > 0:
			logger.error('Invalid file {}:'.format(file))
			logger.error("File_id in {} is different from file_id in submission index file".format(file))
			return False
	return True

def check_index_get_submission_files(ref, subm_dir):

	# Get and check if index file is there
	index_file_path = os.path.join(subm_dir, "system_output.index.tab")
	check_file_exist(index_file_path, subm_dir)

	# Check the format of index file
	column_map = {"index": 4}
	header_map = {"index":{"file_id": "object","is_processed": "bool","message": "object","file_path": "object"}}

	if individual_file_check("index", index_file_path, column_map, header_map, processed_label=None, subm_file=None, length=None, ref_df=None, norm_list=None):
		index_df = pd.read_csv(index_file_path, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

		# Then check if file_id in reference is equal to file_id in index file
		if sorted(list(index_df["file_id"])) != sorted(list(ref["file_id"].unique())):
			logger.error('File_ids in reference are different from file_ids in index_file')
			logger.error('Validation failed')
			exit(1)
	
		# Then check submission path, if it's ok, append path into dictionary
		subm_file_paths_dict = {}
		for j in index_df["file_id"]:
			processed_label = index_df["is_processed"][index_df["file_id"] == j].values[0]
			subm_file_path = index_df["file_path"][index_df["file_id"] == j].values[0]
			if subm_file_path != subm_file_path: # check if it's NaN value
				logger.error("Can't find submission file path for {}".format(j))
				logger.error('Validation failed')
				exit(1)

			type = ref["type"][ref["file_id"] == j].values[0]
			length = ref["length"][ref["file_id"] == j].values[0]
			if subm_file_path[:2] == './': #Check if path is start with ./
				subm_file_path = subm_file_path[2:]
			if subm_dir not in subm_file_path: # Check it's absolute or relative path
				subm_file_path = os.path.join(subm_dir, subm_file_path)

			check_file_exist(subm_file_path, subm_dir)

			subm_file_paths_dict[j] = {"path": subm_file_path, "type": type, "processed": processed_label, "length": length}
	else:
		logger.error('Validation failed')
		exit(1)

	return index_file_path, subm_file_paths_dict

def check_start_end_timestamp_within_length(file, task, length):

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	if df.shape[0] != 0:
		invalid_point = []

		if task == "changepoint":
			for i in range(df.shape[0]):
				if df.iloc[i]["timestamp"] > length:
					invalid_point.append(df.iloc[i]["timestamp"])
			
		else:
			for i in range(df.shape[0]):
				if df.iloc[i]["start"] >= length:
					invalid_point.append(df.iloc[i]["start"])
				if df.iloc[i]["end"] > length:
					invalid_point.append(df.iloc[i]["end"])
			
		if len(invalid_point) > 0:
			logger.error('Invalid file {}:'.format(file))
			logger.error("Start/end/timestamp in {} is not within the file length".format(file))
			return False
	return True

def check_value_range(file, task):

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	if df.shape[0] != 0:
		invalid_value_range = []
		for i in df[task]:
			if not((i >= 1) and (i <= 1000)):
				invalid_value_range.append(i)

		if len(invalid_value_range) > 0:
			logger.error('Invalid file {}:'.format(file))
			logger.error("{} in {} is not in the range of [1,1000]".format(task, file))
			return False
	return True

def check_ref_norm(hidden_norm, mapping_file):

	mapping_df = pd.read_csv(mapping_file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	invalid_ref_norm = []
	for i in mapping_df["ref_norm"]:
		if i not in hidden_norm:
			invalid_ref_norm.append(i)
	if len(invalid_ref_norm) > 0:
		logger.error('Validation failed')
		logger.error("Additional ref_norm '{}'' have been found in mapping file".format(set(invalid_ref_norm)))
		return False
	return True

def extract_modality_info(file_type):

	if file_type == "text":
		frame_data_type = "int"
	else:
		frame_data_type = "float"

	column_map = {"norms": 6, "emotions": 5, "valence_continuous": 4, "arousal_continuous": 4, "changepoint": 3}
	header_map = {"norms":{"file_id": "object","norm": "object","start": frame_data_type,"end": frame_data_type,"status": "object","llr": "float"},
				"emotions":{"file_id": "object","emotion": "object","start": frame_data_type,"end": frame_data_type,"llr": "float"},
				"valence_continuous":{"file_id": "object","start": frame_data_type,"end": frame_data_type,"valence_continuous": "int"},
				"arousal_continuous":{"file_id": "object","start": frame_data_type,"end": frame_data_type,"arousal_continuous": "int"},
				"changepoint":{"file_id": "object","timestamp": frame_data_type,"llr": "float"}}

	return column_map, header_map