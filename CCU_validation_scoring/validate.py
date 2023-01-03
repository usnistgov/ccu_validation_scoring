import os, glob
import pandas as pd
from pathlib import Path
from .preprocess_reference import *
import logging

logger = logging.getLogger('VALIDATION')

def global_ref_file_checks(file, index_dir):
	
	if file != None and "valence_arousal.tab" in file:
		file_type = "valence"
		expected_header = ['user_id', 'file_id', 'segment_id', 'valence_continuous', 'valence_binned', 'arousal_continuous', 'arousal_binned']
		expected_column_number = len(expected_header)
		expected_header_types = {'user_id': 'int', 'file_id': 'object', 'segment_id':'object'}

	elif file != None and "changepoint.tab" in file:
		file_type = "changepoint"
		expected_header = ['user_id', 'file_id', 'timestamp', 'impact_scalar', 'comment']
		expected_column_number = len(expected_header)
		expected_header_types = {'user_id': 'int', 'file_id': 'object', 'timestamp': ["int", "float"], 'impact_scalar':'int', 'comment': 'object'}

	elif file != None and "emotions.tab" in file: # CHANGE TO emotions.tab
		file_type = "emotions"
		expected_header = ['user_id', 'file_id', 'segment_id', 'emotion', 'multi_speaker']
		expected_column_number = len(expected_header)
		expected_header_types = {'user_id': 'int', 'file_id': 'object', 'segment_id': 'object', 'emotion': 'object', 'multi_speaker': 'object'}

	elif file != None and "norms.tab" in file:
		file_type = "norms"
		expected_header = ['user_id', 'file_id', 'segment_id', 'norm',	'status']
		expected_column_number = len(expected_header)
		expected_header_types = {'user_id': 'int', 'file_id': 'object', 'segment_id': 'object', 'norm': 'object', 'status': 'object'}

	elif file != None and "segments.tab" in file:
		file_type = "segments"
		expected_header = ['file_id', 'segment_id', 'start', 'end']
		expected_column_number = len(expected_header)
		expected_header_types = {'file_id': 'object', 'segment_id': 'object', 'start': ["int", "float"], 'end': ["int", "float"]}

	file_checks = (check_valid_tab(file) and
	check_column_number(file, expected_column_number) and
	check_valid_header(file,expected_header) and
	check_ref_file_data_types(file, expected_header_types, file_type) and
	check_ref_file_ids(file, index_dir))

	return file_checks

def global_file_checks(reference_dir, submission_dir):
	
	index_file_path, subm_file_dict = check_index_get_submission_files(reference_dir, submission_dir)
	check_submission_files(submission_dir, index_file_path, subm_file_dict)

	return subm_file_dict

def individual_file_check(task, type, subm_file_path, column_map, header_map, processed_label, subm_file, length, norm_list):
	
	if task == "norms":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, task, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_start_small_end(subm_file_path, type) and
			check_start_end_timestamp_within_length(subm_file_path, task, length, type))
	
	if task == "emotions":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, task, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_emotion_id(subm_file_path) and
			check_start_small_end(subm_file_path, type) and
			check_start_end_timestamp_within_length(subm_file_path, task, length, type))

	if task == "valence_continuous" or task == "arousal_continuous":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, task, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_start_small_end(subm_file_path, type) and
			check_time_no_gap(subm_file_path, type) and
			check_duration_cover(subm_file_path, length) and
			check_value_range(subm_file_path, task) and
			check_start_end_timestamp_within_length(subm_file_path, task, length, type))

	if task == "changepoint":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, task, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_start_end_timestamp_within_length(subm_file_path, task, length, type))

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

		# Check whether we had too many docs in the submission files
		invalid_file = set(subm_files) - set([i["path"] for i in subm_file_dict.values()])
		if invalid_file:
			logger.error("Additional file(s) {} have been found in submission {}".format(invalid_file, subm_dir))
 
		# Check whether we had too many docs in the reference files
		invalid_file = set([i["path"] for i in subm_file_dict.values()]) - set(subm_files)
		if invalid_file:
			logger.error('The following document(s) {} were not found: {}'.format(invalid_file, subm_dir))
			
		logger.error('Validation failed')
		exit(1)
	else:
		pass

	return None

def check_file_exist(file_path, file, dir):

	if len(glob.glob(file_path)) >= 1:
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

def check_output_records(file, task, processed_label):

	df = pd.read_csv(file, dtype={'norm': object}, sep='\t')

	if task == "valence_continuous" or task == "arousal_continuous":
		if processed_label == False and df.shape[0] != 0:
			logger.error("Output records have been found in submission file {} with False is_processed label".format(file))
			return False
		if processed_label == True and df.shape[0] == 0:
			logger.error("Can't find output records in vd/ad submission file {} with True is_processed label".format(file))
			return False
	else:
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
			if type(header_type[i]) is list:		
				if res[i] not in header_type[i]:
					invalid_type_column.append(i)
			else:
				if res[i] != header_type[i]:
					invalid_type_column.append(i)

		if len(invalid_type_column) > 0:
			logger.error('Invalid file {}:'.format(file))
			logger.error("The data type of column {} in file {} is invalid".format(invalid_type_column, file))
			return False
	return True

def check_emotion_id(file):

	df = pd.read_csv(file, sep='\t')
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

def check_start_small_end(file, type):

	df = pd.read_csv(file, dtype={'norm': object}, sep='\t')
	if df.shape[0] != 0:
		if type == "text":	
			for i in range(df.shape[0]):
				if df.iloc[i]["start"] > df.iloc[i]["end"]:
					logger.error('Invalid file {}:'.format(file))
					logger.error("Start is higher than end in text {}".format(file))
					return False
		if type == "audio" or type == "video":	
			for i in range(df.shape[0]):
				if df.iloc[i]["start"] >= df.iloc[i]["end"]:
					logger.error('Invalid file {}:'.format(file))
					logger.error("Start is equal to/higher than end in audio/video {}".format(file))
					return False
	return True

def check_time_no_gap(file, type):

	df = pd.read_csv(file, sep='\t')
	if df.shape[0] != 0:
		df_sorted = df.sort_values(by=['start','end'])

		if type == "text":
			for i in range(df_sorted.shape[0]-1):
				if df_sorted.iloc[i]["end"] + 1 != df_sorted.iloc[i+1]["start"]:
					logger.error('Invalid file {}:'.format(file))
					logger.error("There are some gaps/overlaps in timestamp of {}".format(file))
					return False
			return True
		
		if type == "audio" or type == "video":
			for i in range(df_sorted.shape[0]-1):
				if abs(df_sorted.iloc[i]["end"] - df_sorted.iloc[i+1]["start"]) >= 0.02:
					logger.error('Invalid file {}:'.format(file))
					logger.error("There are some gaps in timestamp of {}".format(file))
					return False
	return True

def check_duration_cover(file, length):

	df = pd.read_csv(file, dtype={'norm': object}, sep='\t')
	if df.shape[0] != 0:
		start = min(list(df["start"]))
		end = max(list(df["end"]))	

		if start == 0 and end == length:
			return True

		logger.error('Invalid file {}:'.format(file))
		logger.error("The duration of {} is different from the duration of source".format(file))
		return False
	return True

def check_fileid_index_match(file, file_id):

	df = pd.read_csv(file, dtype={'norm': object}, sep='\t')
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

def check_index_get_submission_files(ref_dir, subm_dir):

	# Get and check if index file is there
	system_output_index_file_path = os.path.join(subm_dir, "system_output.index.tab")
	check_file_exist(system_output_index_file_path, system_output_index_file_path, subm_dir)

	system_input_index_file_path = os.path.join(ref_dir, "index_files", "*system_input.index.tab")
	check_file_exist(system_input_index_file_path, system_input_index_file_path, ref_dir)

	file_info_path = os.path.join(ref_dir, "docs", "file_info.tab")
	check_file_exist(file_info_path, file_info_path, ref_dir)

	# Check the format of index file
	column_map = {"index": 4}
	header_map = {"index":{"file_id": "object","is_processed": "bool","message": "object","file_path": "object"}}

	if individual_file_check("index", None, system_output_index_file_path, column_map, header_map, processed_label=None, subm_file=None, length=None, norm_list=None):

		system_output_index_df = pd.read_csv(system_output_index_file_path, sep='\t')
		system_input_index_file_path = glob.glob(system_input_index_file_path)[0]
		system_input_index_df = pd.read_csv(system_input_index_file_path, sep='\t')
		file_info_df = pd.read_csv(file_info_path, sep='\t')
		file_info_df = file_info_df[["file_uid", "type", "length"]]
		file_info_df.drop_duplicates(inplace = True)

		# Then check if file_id in reference is equal to file_id in index file
		if sorted(list(system_output_index_df["file_id"])) != sorted(list(system_input_index_df["file_id"].unique())):
			logger.error('File_ids in system input are different from file_ids in system output')
			logger.error('Validation failed')
			exit(1)
	
		# Then check submission path, if it's ok, append path into dictionary
		subm_file_paths_dict = {}
		for j in system_output_index_df["file_id"]:
			processed_label = system_output_index_df["is_processed"][system_output_index_df["file_id"] == j].values[0]
			subm_file_path = system_output_index_df["file_path"][system_output_index_df["file_id"] == j].values[0]
			if subm_file_path != subm_file_path: # check if it's NaN value
				logger.error("Can't find submission file path for {}".format(j))
				logger.error('Validation failed')
				exit(1)

			type = file_info_df["type"][file_info_df["file_uid"] == j].values[0]
			length = file_info_df["length"][file_info_df["file_uid"] == j].values[0]
			if subm_file_path[:2] == './': #Check if path is start with ./
				subm_file_path = subm_file_path[2:]
			if subm_dir not in subm_file_path: # Check it's absolute or relative path
				full_subm_file_path = os.path.join(subm_dir, subm_file_path)
				check_file_exist(full_subm_file_path, subm_file_path, subm_dir)
				subm_file_paths_dict[j] = {"path": full_subm_file_path, "type": type, "processed": processed_label, "length": length}
			else:
				check_file_exist(subm_file_path, subm_file_path, subm_dir)
				subm_file_paths_dict[j] = {"path": subm_file_path, "type": type, "processed": processed_label, "length": length}
	else:
		logger.error('Validation failed')
		exit(1)

	return system_output_index_file_path, subm_file_paths_dict

def check_start_end_timestamp_within_length(file, task, length, type):

	df = pd.read_csv(file, dtype={'norm': object}, sep='\t')
	if df.shape[0] != 0:
		invalid_point = []

		if task == "changepoint":
			for i in range(df.shape[0]):
				if df.iloc[i]["timestamp"] > length:
					invalid_point.append(df.iloc[i]["timestamp"])

			if len(invalid_point) > 0:
				logger.error('Invalid file {}:'.format(file))
				logger.error("The value of timestamp column in {} is larger than the value of length in file_info.tab".format(file))
				return False
			
		else:
			for i in range(df.shape[0]):
				if type == "text":	
					if df.iloc[i]["start"] > length:
						invalid_point.append(df.iloc[i]["start"])
					if df.iloc[i]["end"] > length:
						invalid_point.append(df.iloc[i]["end"])
				if type == "audio" or type == "video":
					if df.iloc[i]["start"] >= length:
						invalid_point.append(df.iloc[i]["start"])
					if df.iloc[i]["end"] > length:
						invalid_point.append(df.iloc[i]["end"])

			if len(invalid_point) > 0:
				logger.error('Invalid file {}:'.format(file))
				logger.error("The value of start/end column in {} is larger than the value of length in file_info.tab".format(file))
				return False					

	return True

def check_value_range(file, task):

	df = pd.read_csv(file, sep='\t')
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

	mapping_df = pd.read_csv(mapping_file, dtype={'sys_norm': object, 'ref_norm': object}, sep='\t')
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
		frame_data_type = ["int"]
	else:
		frame_data_type = ["int", "float"]

	column_map = {"norms": 6, "emotions": 5, "valence_continuous": 4, "arousal_continuous": 4, "changepoint": 3}
	header_map = {"norms":{"file_id": "object","norm": "object","start": frame_data_type,"end": frame_data_type,"status": "object","llr": "float"},
				"emotions":{"file_id": "object","emotion": "object","start": frame_data_type,"end": frame_data_type,"llr": "float"},
				"valence_continuous":{"file_id": "object","start": frame_data_type,"end": frame_data_type,"valence_continuous": "int"},
				"arousal_continuous":{"file_id": "object","start": frame_data_type,"end": frame_data_type,"arousal_continuous": "int"},
				"changepoint":{"file_id": "object","timestamp": frame_data_type,"llr": "float"}}

	return column_map, header_map

################################################ Reference Directory Validation Checks Below ################################################ 
def check_ref_file_ids(file, index_dir):

	# Collect file ids in system_input.index.tab
	is_directory = os.path.isdir(index_dir)
	if is_directory != True:
		logger.error("Invalid directory: {} is not a directory".format(index_dir))
		exit(1)
	
	file_list = os.listdir(index_dir)
	for file_path in file_list:

		if "system_input.index.tab" in file_path:
			index_file_path = os.path.join(index_dir, file_path)
			index_df = pd.read_csv(index_file_path, sep='\t')

			if index_df.shape[0] != 0:
				file_ids = []
				for index, row in index_df.iterrows():
					if row['file_id'] not in file_ids:
						file_ids.append(row['file_id'])
			
			# Check that file ids in file are same
			invalid_ids =[]
			df = pd.read_csv(file, sep='\t')
			
			if df.shape[0] != 0:
				for index, row in df.iterrows():
					if row['file_id'] not in file_ids:
						invalid_ids.append(row['file_id'])
			
			if len(invalid_ids) > 0:
				logger.error("File {} contains file ids that are not found in file {}: {}".format(file, index_file_path, invalid_ids))
				return False
	
	return True

def check_ref_file_data_types(file, header_types, file_type):
		
	df = pd.read_csv(file, sep='\t')
	df_types = df.dtypes
	
	invalid_type_column = []
	if df.shape[0] != 0:
		for index, row in df.iterrows():
			for column in df.columns.values: # Validate that each value in each column is of the correct type
				if file_type != "valence": # Valence and arousal validation require particular validation checks
					if type(header_types[column]) is list:
						if df_types[column] not in header_types[column]:
							invalid_type_column.append(column)
					else:
						if df_types[column] != header_types[column]:
							invalid_type_column.append(column)
				else: 
					valence_arousal_columns = ["valence_continuous", "valence_binned", "arousal_continuous", "arousal_binned"]
					if column not in valence_arousal_columns:
						if df_types[column] != header_types[column]:
								invalid_type_column.append(column)
					else: 
						if row[column] == "noann" and df_types[column] != 'object':
							invalid_type_column.append(column)
						elif row[column] != "noann" and is_float(row[column]) == False:
							invalid_type_column.append(column)
	
	if len(invalid_type_column) > 0:
		logger.error("Validation failed")
		logger.error("In file {}, the following column(s) has data type errors: {}".format(file, set(invalid_type_column)))
		return False
	
	return True

def check_ref_fileid_segmentid_match(file, docs_dir):
	
	# Read in file ids and segment ids from segments.tab
	segments_file_path = os.path.join(docs_dir, "segments.tab")
	check_file_exist(segments_file_path, segments_file_path, docs_dir)
	segments_df = pd.read_csv(segments_file_path, sep='\t')

	if segments_df.shape[0] != 0:
		expected_file_ids = []
		expected_segment_ids = []
		for index, row in segments_df.iterrows():
			if row['file_id'] not in expected_file_ids:
				expected_file_ids.append(row['file_id'])
			if row['segment_id'] not in expected_segment_ids:
				expected_segment_ids.append(row['segment_id'])
	
	# For each of the other 3 files, loop through and make sure that the file and segment ids are in those lists
	invalid_file_ids = []
	invalid_segment_ids = []
	df = pd.read_csv(file, sep='\t')
	if df.shape[0] != 0:
		for index, row in df.iterrows():
			if row['file_id'] not in expected_file_ids:
				invalid_file_ids.append(row['file_id'])
			if row['segment_id'] not in expected_segment_ids:
				invalid_segment_ids.append(row['segment_id'])
	
		if len(invalid_file_ids) > 0 or len(invalid_segment_ids) > 0:
			logger.error("Invalid file {}".format(file))
			if len(invalid_file_ids) > 0:
				logger.error("File ID(s) {} in {} is not found in {}".format(invalid_file_ids, file, segments_file_path))
				return False
			if len(invalid_segment_ids) > 0:
				logger.error("Segment ID(s) {} in {} is not found in {}".format(invalid_segment_ids, file, segments_file_path))
				return False
	return True

def check_duplicate_emotions(file):

	df = pd.read_csv(file, sep='\t')
	for index, row in df.iterrows():
		emotion_list = [emotion.strip() for emotion in row['emotion'].split(',')] # Convert emotion column to list of individual emotions
		if len(emotion_list) != len(set(emotion_list)):
			logger.error("Validation failed")
			seen = set()
			duplicates = [emotion for emotion in emotion_list if emotion in seen or seen.add(emotion)]
			if duplicates:
				logger.error("File {} contains the following emotion(s) duplicated: {}".format(file, duplicates))
			return False
	return True

def check_valid_emotions(file):
		
	valid_emotions = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation", "none", "noann"]
	df = pd.read_csv(file, sep='\t')

	if df.shape[0] != 0:
		invalid_emotions = []
		for index, row in df.iterrows():
			emotion_list = [emotion.strip() for emotion in row['emotion'].split(',')] # Convert emotion column to list of individual emotions
			for emotion in set(emotion_list):
				if emotion not in valid_emotions:
					invalid_emotions.append(emotion)
		
		if len(invalid_emotions) > 0:
			logger.error("Invalid file {}".format(file))
			logger.error("The following emotions are not valid: {}".format(invalid_emotions))
			return False
	return True

def check_empty_na(file, task):
	
	df = pd.read_csv(file, sep='\t')
	
	if df.shape[0] != 0: 
		invalid_values = []
		for index, row in df.iterrows():
			# Check second to last row (row['emotion'] for emotions.tab and row['norm'])
			if task == "emotions":
				if row["emotion"] == "none" and row['multi_speaker'] != "EMPTY_NA":
					invalid_values.append(row['multi_speaker'])
			if task == "norms":
				if row["norm"] == "none" and row['status'] != "EMPTY_NA":
					invalid_values.append(row['status'])
		
		if len(invalid_values) > 0:
			logger.error("Validation failed {}".format(file))
			logger.error("The following values should have been 'EMPTY_NA': {}".format(invalid_values))
			return False
	return True

def check_norm_range(file):
		
	df = pd.read_csv(file, sep='\t')
	valid_norms = ['none', 'noann'] #subject to change	
	
	if df.shape[0] != 0:
		invalid_norms = []
		for norm in df['norm']:
			if norm not in valid_norms and len(norm) != 3:
				invalid_norms.append(norm)

		if len(invalid_norms) > 0:
			logger.error("Invalid file {}".format(file))
			logger.error("Invalid norm IDs '{}' have been found in {} ".format(set(invalid_norms), file))
			return False
	
	return True

def check_noann_all_columns(file):

	df = pd.read_csv(file, sep='\t')

	if df.shape[0] != 0: 
		for index, row in df.iterrows():
			if file != None and "valence" in file:
				val_and_arousal_tags = [row['valence_continuous'], row['valence_binned'], row['arousal_continuous'], row['arousal_binned']]
				if "noann" in val_and_arousal_tags:
					noann_count = val_and_arousal_tags.count("noann")
					if noann_count != 4:
						logger.error("Validation failed in file {}".format(file))
						logger.error("Expected 'noann' tags in for all valence/arousal columns")
						return False
			elif file != None and "emotion" in file:
				emotion_and_multi_tags = [row['emotion'], row['multi_speaker']]
				if "noann" in emotion_and_multi_tags:
					noann_count = emotion_and_multi_tags.count("noann")
					if noann_count != 2:
						logger.error("Validation failed in file {}".format(file))
						logger.error("Expected 'noann' tags for both the emotion and the multi_speaker columns")
						return False
	return True
					
def check_noann_all_annotators(file):
		
	# Make sure "noann" tags applied to all value columns
	columns_valid = check_noann_all_columns(file)

	if columns_valid != True:
		return False
	
	df = pd.read_csv(file, sep='\t')
	
	if df.shape[0] != 0: 
		last_segment_id = ""
		for index, row in df.iterrows():
			# Create a partial dataframe containing only rows w/ this specific segment_id
			current_segment_id = row['segment_id']
			if current_segment_id != last_segment_id:
				partial_df = df[df['segment_id'] == current_segment_id]
			else:
				continue

			# Check if this segment_id has any "noann" values. If so, make sure that all annotators said no speech
			noann_df = partial_df[partial_df['valence_continuous'] == "noann"]
			if noann_df.shape[0] == 0:
				last_segment_id = current_segment_id
				continue
			else: 
				# Retrieve user_ids of those who did not tag noann
				missing_noann_df = partial_df[partial_df['valence_continuous'] != "noann"]
				if missing_noann_df.shape[0] != 0:
					logger.error("Invalid file: {}".format(file))
					logger.error("Inconsistent 'noann' tags in segment {}".format(current_segment_id))
					logger.error("The following annotators tagged 'noann': {}".format(set(noann_df['user_id'])))
					logger.error("The following annotators did not tag 'noann': {}".format(set(missing_noann_df['user_id'])))
					return False

			last_segment_id = current_segment_id
	return True

def check_valence_arousal_range(file):
	
	df = pd.read_csv(file, sep='\t')
	
	valence_arousal_columns = ['valence_continuous', 'valence_binned', 'arousal_continuous', 'arousal_binned']
	
	if df.shape[0] != 0:
		invalid_ranges = []
		for index, row in df.iterrows():
			for column in valence_arousal_columns:
				if row[column] != "noann":
					if not (float(row[column]) >= 1 and float(row[column]) <= 1000):
						invalid_ranges.append(row[column])

		if len(invalid_ranges) > 0:
			logger.error("Invalid file {}".format(file))
			logger.error("The following values are out of range: {}".format(invalid_ranges))
			return False
	return True

def check_start_end_types(file, docs_dir):

	# Open system input file and record which files are text and which files are audio/video using two lists
	file_info_path = os.path.join(docs_dir, "file_info.tab") 
	check_file_exist(file_info_path, file_info_path, docs_dir)
	file_info_df = pd.read_csv(file_info_path, sep='\t')

	if file_info_df.shape[0] != 0:
		text_ids = []
		audvid_ids = []
		for index, row in file_info_df .iterrows():
			if row['type'] == "text":
				text_ids.append(row['file_uid']) # Since file_id column does not exist, I used file_uid. Switch to original_file_id?
			elif row['type'] == "audio" or row['type'] == "video":
				audvid_ids.append(row['file_uid'])
			else:
				continue
	
	# Open segments.tab and for every file_id, if it's in one list, make sure it has a data type of ___ and vice versa otherwise
	segments_df = pd.read_csv(file, sep='\t')
	
	if segments_df.shape[0] != 0:
		invalid_types = []
		for index, row in segments_df.iterrows():
			if row['file_id'] in text_ids:
				if float(row['start']).is_integer() == False or float(row['end']).is_integer() == False:
					invalid_types.append(row['start'])
					invalid_types.append(row['end'])
			elif row['file_id'] in audvid_ids:
				if is_float(row['start']) == False or is_float(row['end']) == False:
					invalid_types.append(row['start'])
					invalid_types.append(row['end'])
			else:
				logger.error("Validation failed")
				logger.error("Type of file ID {} in file {} is not text, audio, or video".format(row['file_id'], file))
				return False

		if len(invalid_types) > 0:
			logger.error("Validation failed")
			logger.error("In file {}, the following start/end times are the wrong data type: {}".format(file, invalid_types))
			return False
	return True

def check_valid_timestamps(file, docs_dir):
	# Open system input file and record which files are text and which files are audio/video using two lists
	file_info_path = os.path.join(docs_dir, "file_info.tab") 
	check_file_exist(file_info_path, file_info_path, docs_dir)
	file_info_df = pd.read_csv(file_info_path, sep='\t')

	if file_info_df .shape[0] != 0:
		text_ids = []
		audvid_ids = []
		for index, row in file_info_df.iterrows():
			if row['type'] == "text":
				text_ids.append(row['file_uid']) # Since file_id column does not exist, I used file_uid. Switch to original_file_id?
			elif row['type'] == "audio" or row['type'] == "video":
				audvid_ids.append(row['file_uid'])
			else:
				continue
		
	# Open segments.tab and for every file_id, if it's in one list, make sure it has a data type of ___ and vice versa otherwise
	segments_df = pd.read_csv(file, sep='\t')

	if segments_df.shape[0] != 0:
		for index, row in segments_df.iterrows():
			if row['file_id'] in text_ids:
				if row['start'] > row['end']:
					logger.error('Invalid file {}:'.format(file))
					logger.error("Start is higher than end in text {}".format(file))
					return False
			elif row['file_id'] in audvid_ids:
				if row['start'] >= row['end']:
					logger.error('Invalid file {}:'.format(file))
					logger.error("Start is equal to/higher than end in audio/video {}".format(file))
					return False
			else:
				logger.error("Validation failed")
				logger.error("Type of file ID {} in file {} is not text, audio, or video".format(row['file_id'], file))
				return False
	return True

	