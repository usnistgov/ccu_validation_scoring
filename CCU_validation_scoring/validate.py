from functools import partial
import os
import pandas as pd
from pathlib import Path
from .preprocess_reference import *
import logging

logger = logging.getLogger('VALIDATION')

def global_file_checks(task, reference_dir, submission_dir, scoring_index_file):
    
	try:
		scoring_index = pd.read_csv(scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('{} is not a valid scoring index file'.format(scoring_index_file))
		exit(1)

	ref = preprocess_reference_dir(ref_dir = reference_dir, scoring_index = scoring_index, task = task)
	index_file_path, subm_file_dict = check_index_get_submission_files(ref, submission_dir)
	check_submission_files(submission_dir, index_file_path, subm_file_dict)

	return ref, subm_file_dict

def validate_ref_individial_file_check(task, ref_file_path, column_map, header_map, ref_file):
	if task == "norms":
		file_checks = (check_valid_tab(ref_file_path) and
			check_column_number(ref_file_path,column_map[task]) and
			check_valid_header(ref_file_path,list(header_map[task])) and
			#check_output_records(subm_file_path, task, processed_label) and
			#check_data_type(subm_file_path, header_map[task]) and 
			check_fileid_index_match(ref_file_path, ref_file) and 
			check_norm_range(ref_file_path) )
	
	if task == "emotions":
		file_checks =(check_valid_tab(ref_file_path) and
			check_column_number(ref_file_path,column_map[task]) and
			check_valid_header(ref_file_path,list(header_map[task])) and
			#check_output_records(subm_file_path, task, processed_label) and
			#check_data_type(subm_file_path, header_map[task]) and 
			check_fileid_index_match(ref_file_path, ref_file) and
			check_emotion_id(ref_file_path) and 
			check_duplicate_emotions(ref_file_path) and
			check_empty_na(task, ref_file_path))

	if task == "valence_continuous" or task == "arousal_continuous":
		file_checks =(check_valid_tab(ref_file_path) and
			check_column_number(ref_file_path,column_map[task]) and
			check_valid_header(ref_file_path,list(header_map[task])) and
			#check_output_records(subm_file_path, task, processed_label) and
			#check_data_type(subm_file_path, header_map[task]) and 
			check_fileid_index_match(ref_file_path, ref_file) and 
			check_value_range(ref_file_path, task) and
			check_nospeech_all_columns(ref_file_path) and 
			check_nospeech_all_annotators(ref_file_path))	

	if task == "segment":
		file_checks =(check_valid_tab(ref_file_path) and
			check_column_number(ref_file_path,column_map[task]) and
			check_valid_header(ref_file_path,list(header_map[task])) and
			#check_output_records(subm_file_path, task, processed_label) and
			#check_data_type(subm_file_path, header_map[task]) and 
			check_fileid_index_match(ref_file_path, ref_file) and 
			check_start_small_end(ref_file_path))

def individual_file_check(task, subm_file_path, column_map, header_map, processed_label, subm_file, length, ref_df, norm_list):
	
	if task == "norms":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, task, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_start_small_end(subm_file_path) and
			check_start_end_timestamp_within_length(subm_file_path, task, length))
	
	if task == "emotions":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, task, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_emotion_id(subm_file_path) and
			check_start_small_end(subm_file_path) and
			check_start_end_timestamp_within_length(subm_file_path, task, length))

	if task == "valence_continuous" or task == "arousal_continuous":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, task, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_start_small_end(subm_file_path) and
			check_time_no_gap(subm_file_path, ref_df) and
			check_duration_equal(subm_file_path, ref_df) and
			check_start_end_timestamp_within_length(subm_file_path, task, length) and
			check_value_range(subm_file_path, task))

	if task == "changepoint":
		file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, task, processed_label) and
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

	if os.path.exists(file_path):
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

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

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
				logger.error("Start is equal to /higher than end in {}".format(file))
				return False
	return True

def check_time_no_gap(file, ref):

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	df_sorted = df.sort_values(by=['start','end'])

	if ref["type"].unique()[0] == "text":
		for i in range(df_sorted.shape[0]-1):
			if df_sorted.iloc[i]["end"] + 1 != df_sorted.iloc[i+1]["start"]:
				logger.error('Invalid file {}:'.format(file))
				logger.error("There are some gaps in timestamp of {}".format(file))
				return False
		return True
	
	if ref["type"].unique()[0] == "audio" or ref["type"].unique()[0] == "video":
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

	if df.shape[0] != 0:
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
	check_file_exist(index_file_path, index_file_path, subm_dir)

	# Check the format of index file
	column_map = {"index": 4}
	header_map = {"index":{"file_id": "object","is_processed": "bool","message": "object","file_path": "object"}}

	if individual_file_check("index", index_file_path, column_map, header_map, processed_label=None, subm_file=None, length=None, ref_df=None, norm_list=None):
		index_df = pd.read_csv(index_file_path, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

		# Then check if file_id in reference is equal to file_id in index file
		if sorted(list(index_df["file_id"])) != sorted(list(ref["file_id"].unique())):
			logger.error('File_ids in index_file are different from file_ids in reference')
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
				full_subm_file_path = os.path.join(subm_dir, subm_file_path)
				check_file_exist(full_subm_file_path, subm_file_path, subm_dir)
				subm_file_paths_dict[j] = {"path": full_subm_file_path, "type": type, "processed": processed_label, "length": length}
			else:
				check_file_exist(subm_file_path, subm_file_path, subm_dir)
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
	
def check_duplicate_emotions(file):
	
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	for index, row in df.iterrows():
		emotion_list = row['emotion'].split()
		uniq_emotions = set(emotion_list)
		if len(emotion_list) != len(uniq_emotions):
			logger.error("Validation failed")
			seen = set()
			duplicates = [emotion for emotion in emotion_list if emotion in seen or seen.add(emotion)]
			if duplicates:
				logger.error("File {} contains the following emotion(s) duplicated: {}".format(file, duplicates))

def check_empty_na(task, file):
	
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
		
	for index, row in df.iterrows():
		# Check second to last row (row['emotion'] for emotions.tab and row['norm'])
		if task == "emotions":
			if row["emotion"] == "none" and row['multi_speaker'] != "EMPTY_NA":
				logger.error("Validation failed {}".format(file))
				logger.error("Expected 'EMPTY_NA' status, got {}".format(row['multi_speaker']))
		if task == "norms":
			if row["norm"] == "none" and row['status'] != "EMPTY_NA":
				logger.error("Validation failed {}".format(file))
				logger.error("Expected 'EMPTY_NA' status, got {}".format(row['status']))

def check_nospeech_all_columns(file):
		
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	for index, row in df.iterrows():
		val_and_arousal_tags = [row['valence_continuous'], row['valence_binned'], row['arousal_continuous'], row['arousal_binned']]
		if "nospeech" in val_and_arousal_tags:
			nospeech_count = val_and_arousal_tags.count("nospeech")
			if nospeech_count != 4:
				logger.error("Validation failed")
				logger.error("Expected 4 'nospeech' tags but only got {} in file {}".format(nospeech_count, file))
					
def check_nospeech_all_annotators(file):
    	
	# Make sure "nospeech" tags applied to all value columns
	check_nospeech_all_columns(file)
	
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	
	last_segment_id = ""
	for index, row in df.iterrows():
		# Create a partial dataframe containing only rows w/ this specific segment_id
		current_segment_id = row['segment_id']
		if current_segment_id != last_segment_id:
			partial_df = df[df['segment_id'] == current_segment_id]
		else:
			continue

		# Check if this segment_id has any "nospeech values". If so, make sure that all annotators said no speech
		nospeech_df = partial_df[partial_df['valence_continuous'] == "nospeech"]
		if nospeech_df.shape[0] == 0:
			last_segment_id = current_segment_id
			continue
		else: 
			# Retrieve user_ids of those who did not tag nospeech
			missing_nospeech_df = partial_df[partial_df['valence_continuous'] != "nospeech"]
			if missing_nospeech_df.shape[0] != 0:
				logger.error("Validation failed")
				logger.error("Inconsistent 'nospeech' tags in segment {}".format(current_segment_id))
				logger.error("The following annotators tagged 'nospeech': {}".format(set(nospeech_df['user_id'])))
				logger.error("The following annotators did not tag 'nospeech': {}".format(set(missing_nospeech_df['user_id'])))

		last_segment_id = current_segment_id

def check_norm_range(file):
	
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	valid_norms = ['001', '002', '007', 'none', 'nospeech'] #subject to change	
	if df.shape[0] != 0:
		invalid_norms = []
		for norm in df['norm']:
			if norm not in valid_norms:
				invalid_norms.append(norm)

		if len(invalid_norms) > 0:
			logger.error("Invalid file {}", file)
			logger.error("Additional emotion(s) '{}' have been found in {} ".format(set(invalid_norms), file))
			return False
	
	return True

def check_start_end_types(file, ref_dir):

	# Open system input file and record which files are text and which files are audio/video using two lists
	input_file_path = os.path.join(ref_dir, "system_input.index.tab")
	check_file_exist(input_file_path, input_file_path, ref_dir)
	input_df = pd.read_csv(input_file_path, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

	text_ids = []
	audvid_ids = []
	for index, row in input_df.iterrows():
		if row['type'] == "text":
			text_ids.append(row['file_id'])
		elif row['type'] == "audio" or row['type'] == "video":
			audvid_ids.append(row['file_id'])
		else:
			logger.error("Invalid file")
			logger.error("File {} has a type other than text, audio, or video for file_id {}".format(input_file_path, row['file_id']))
	
	# Open segments.tab and for every file_id, if it's in one list, make sure it has a data type of ___ and vice versa otherwise
	segments_df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	for index, row in segments_df.iterrows():
		if row['file_id'] in text_ids:
			if row['start'].is_integer() == row['end'].is_integer() == False:
				logger.error("Validation failed")
				logger.error("In file {}, file id {} contains start/end time that are not ints".format(file, row['file_id']))
		elif row['file_id'] in audvid_ids:
			if isinstance(row['start'], float) == isinstance(row['end'], float) == False:
				logger.error("Validation failed")
				logger.error("In file {}, file id {} contains start/end time that are not floats".format(file, row['file_id']))
		else:
			logger.error("Validation failed")
			logger.error("Type of file ID {} in file {} is not text, audio, or video".format(row['file_id'], file))

def check_fileid_segmentid_match(file, ref_dir):
	# Read in file ids and segment ids from segments.tab
	segments_file_path = os.path.join(ref_dir, "segments.tab")
	check_file_exist(segments_file_path, segments_file_path, ref_dir)
	segments_df = pd.read_csv(segments_file_path, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

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
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
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
			if len(invalid_segment_ids) > 0:
				logger.error("Segment ID(s) {} in {} is not found in {}".format(invalid_segment_ids, file, segments_file_path))
