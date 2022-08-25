import os
import pandas as pd
from pathlib import Path
from .preprocess_reference import *
import logging

logger = logging.getLogger('VALIDATION')

def check_submission_files(subm_dir, index_file_path, index_df, subm_file_dict):

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
			logger.error("Additional file(s) {} have been found in submission {}".format(invalid_file, index_file_path))
			
		logger.error('Validation failed')
		exit(1)
	else:
		pass

	return None

def add_length_subm_file_dict(ref_dir, subm_file_dict):

	index_file = os.path.join(ref_dir,"docs","system_input.index.tab")
	index_df = read_dedupe_reference_file(index_file)

	for file_id in subm_file_dict:
		try:
			length = index_df["length"][index_df["file_id"] == file_id].values[0]
			subm_file_dict[file_id]["length"] = length
		except Exception as e:
			logger.error("Can't find corresponding length of {} in system input index file".format(file_id))
			exit(1)
	
	return subm_file_dict

def check_valid_tab(subm_file_path):

	try:
		pd.read_csv(subm_file_path, dtype={'norm': object}, sep='\t')
		return True
	except Exception as e:
		logger.error('{} is not a valid tab file'.format(subm_file_path))
		return False

def check_column_number(subm_file_path, columns_number):

	submission_df = pd.read_csv(subm_file_path, dtype={'norm': object}, sep='\t')

	if submission_df.shape[1] != columns_number:
		logger.error('Invalid file {}:'.format(subm_file_path))
		logger.error('File {} should contain {} columns.'.format(subm_file_path, columns_number))
		return False
	return True

def check_valid_header(subm_file_path, header):

	submission_df = pd.read_csv(subm_file_path, dtype={'norm': object}, sep='\t')

	if set(list(submission_df.columns)) != set(header):
		logger.error('Invalid file {}:'.format(subm_file_path))

		# Check if we had not enough docs in the  reference files
		invalid_header = set(header) - set(list(submission_df.columns))
		if invalid_header:
			logger.error("Header of '{}' is missing following fields: {}".format(subm_file_path, invalid_header))
 
		# Check whether we had too many docs in the reference files
		invalid_header = set(list(submission_df.columns)) - set(header)
		if invalid_header:
			logger.error("Additional field(s) '{}'' have been found in header of {}".format(invalid_header, subm_file_path))

		return False
	return True

def check_output_records(subm_file_path, processed_label):

	submission_df = pd.read_csv(subm_file_path, dtype={'message': object, 'norm': object}, sep='\t')
	if processed_label == False and submission_df.shape[0] != 0:
		logger.error("Output records have been found in submission file with False is_processed label")
		return False
	return True

def check_data_type(subm_file_path, header_type):

	submission_df = pd.read_csv(subm_file_path, dtype={'message': object, 'norm': object}, sep='\t')
	if submission_df.shape[0] != 0:
		res = submission_df.dtypes
		invalid_type_column = []
		for i in submission_df.columns:
			if res[i] != header_type[i]:
				invalid_type_column.append(i)

		if len(invalid_type_column) > 0:
			logger.error('Invalid file {}:'.format(subm_file_path))
			logger.error("The data type of column {} in file {} is invalid".format(invalid_type_column, subm_file_path))
			return False
	return True

def check_emotion_id(subm_file_path):

	submission_df = pd.read_csv(subm_file_path, sep='\t')
	if submission_df.shape[0] != 0:
		invalid_emotion = []
		for i in submission_df["emotion"]:
			if i not in ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]:
				invalid_emotion.append(i)

		if len(invalid_emotion) > 0:
			logger.error('Invalid file {}:'.format(subm_file_path))
			logger.error("Additional emotion(s) '{}'' have been found in {}".format(set(invalid_emotion), subm_file_path))
			return False
	return True

def check_start_small_end(subm_file_path):

	submission_df = pd.read_csv(subm_file_path, dtype={'norm': object}, sep='\t')
	if submission_df.shape[0] != 0:
		for i in range(submission_df.shape[0]):
			if submission_df.iloc[i]["start"] >= submission_df.iloc[i]["end"]:
				logger.error('Invalid file {}:'.format(subm_file_path))
				logger.error("Start is equal to /larger than end in {}".format(subm_file_path))
				return False
	return True

def check_time_no_gap(subm_file_path, header_type):

	submission_df = pd.read_csv(subm_file_path, dtype={'norm': object}, sep='\t')

	if header_type["start"] == "int":
		for i in range(submission_df.shape[0]-1):
			if submission_df.iloc[i]["end"] + 1 != submission_df.iloc[i+1]["start"]:
				logger.error('Invalid file {}:'.format(subm_file_path))
				logger.error("There are some gaps in timestamp of {}".format(subm_file_path))
				return False
		return True
	
	if header_type["start"] == "float":
		for i in range(submission_df.shape[0]-1):
			if submission_df.iloc[i]["end"] != submission_df.iloc[i+1]["start"]:
				logger.error('Invalid file {}:'.format(subm_file_path))
				logger.error("There are some gaps in timestamp of {}".format(subm_file_path))
				return False
		return True

def check_duration_equal(subm_file_path, ref_file_path):

	submission_df = pd.read_csv(subm_file_path, dtype={'norm': object}, sep='\t')
	reference_df = pd.read_csv(ref_file_path, sep='\t')

	def calculate_duration_vd_ad(df):
		start = list(df["start"])
		end = list(df["end"])
		time_pool = start + end
		duration = max(time_pool)-min(time_pool)

		return duration

	if calculate_duration_vd_ad(submission_df) != calculate_duration_vd_ad(reference_df):
		logger.error('Invalid file {}:'.format(subm_file_path))
		logger.error("The duration of {} is different from the duration of {}".format(subm_file_path, ref_file_path))
		return False

	return True

def check_fileid_index_match(subm_file_path, file_id):

	submission_df = pd.read_csv(subm_file_path, dtype={'norm': object}, sep='\t')
	if submission_df.shape[0] != 0:
		invalid_fileid = []
		for i in submission_df["file_id"]:
			if i != file_id:
				invalid_fileid.append(i)

		if len(invalid_fileid) > 0:
			logger.error('Invalid file {}:'.format(subm_file_path))
			logger.error("File_id in {} is different from file_id in submission index file".format(subm_file_path))
			return False
	return True

def check_index_get_submission_files(ref, subm_dir):

	# Get and check if index file is there
	index_file_path = os.path.join(subm_dir, "system_output.index.tab")
	if os.path.exists(index_file_path):
		pass
	else:
		logger.error('No index file found in {}'.format(index_file_path))
		logger.error('Validation failed')
		exit(1)

	index_map = {"file_id": "object","is_processed": "bool","message":"object","file_path":"object"}

	# Check the format of index file
	if (check_valid_tab(index_file_path) and
		check_column_number(index_file_path,4) and
		check_valid_header(index_file_path,list(index_map)) and
		check_data_type(index_file_path, index_map)):

		index_df = pd.read_csv(index_file_path, sep='\t')

		# Then check if file_id in reference is equal to file_id in index file
		if sorted(list(index_df["file_id"])) != sorted(list(ref["file_id"].unique())):
			logger.error('File_ids in reference are different from file_ids in index_file')
			logger.error('Validation failed')
			exit(1)
	
		# Then check submission path, if it's ok, 
		subm_file_paths_dict = {}
		for j in index_df["file_id"]:
			processed_label = index_df["is_processed"][index_df["file_id"] == j].values[0]
			subm_file_path = index_df["file_path"][index_df["file_id"] == j].values[0]
			if subm_file_path != subm_file_path: # check if it's NaN value
				logger.error("Can't find submission file path for {}".format(j))
				logger.error('Validation failed')
				exit(1)

			type = ref["type"][ref["file_id"] == j].values[0]
			if subm_file_path[:2] == './': #Check if path is start with ./
				subm_file_path = subm_file_path[2:]
			if subm_dir not in subm_file_path: # Check it's absolute or relative path
				subm_file_path = os.path.join(subm_dir, subm_file_path)

			if os.path.exists(subm_file_path): # Make sure each path is accessible
				subm_file_paths_dict[j] = {"path": subm_file_path, "type": type, "processed": processed_label}
			else:
				logger.error("Submission file path of file {} is inaccessible".format(j))
				logger.error('Validation failed')
				exit(1)
	else:
		logger.error('Validation failed')
		exit(1)

	return index_file_path, index_df, subm_file_paths_dict

def check_start_end_within_length(subm_file_path, length):

	submission_df = pd.read_csv(subm_file_path, dtype={'norm': object}, sep='\t')
	if submission_df.shape[0] != 0:
		invalid_start_end = []
		for i in range(submission_df.shape[0]):
			if submission_df.iloc[i]["start"] >= length:
				invalid_start_end.append(submission_df.iloc[i]["start"])
			if submission_df.iloc[i]["end"] > length:
				invalid_start_end.append(submission_df.iloc[i]["end"])
		
		if len(invalid_start_end) > 0:
			logger.error('Invalid file {}:'.format(subm_file_path))
			logger.error("Start/end in {} is not within the file length".format(subm_file_path))
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