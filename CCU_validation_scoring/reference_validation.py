import os
import pandas as pd
from utils import *
from pathlib import Path
from .preprocess_reference import *
from .validate import *
import logging

logger = logging.getLogger('VALIDATION')

def global_checks(file, index_dir):

	if file != None and "valence_arousal.tab" in file:
		file_type = "valence"
		expected_header = ['user_id', 'file_id', 'segment_id', 'valence_continuous', 'valence_binned', 'arousal_continuous', 'arousal_binned']
		expected_column_number = len(expected_header)
		expected_header_types = {'user_id': 'int', 'file_id': 'object', 'segment_id':'object'}

	elif file != None and "changepoint.tab" in file:
		file_type = "changepoint"
		expected_header = ['user_id', 'file_id', 'timestamp', 'impact_scalar', 'comment']
		expected_column_number = len(expected_header)
		expected_header_types = {'user_id': 'int', 'file_id': 'object', 'timestamp': 'float', 'impact_scalar':'int', 'comment': 'object'}

	elif file != None and "emotions.tab" in file:
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
		expected_header_types = {'file_id': 'object', 'segment_id': 'object', 'start': 'float', 'end': 'float'}

	file_checks = (check_valid_tab(file) and
	check_column_number(file, expected_column_number) and
	check_valid_header(file,expected_header) and
	check_file_data_types(file, expected_header_types, file_type) and
	check_file_ids(file, index_dir))

	return file_checks

def check_file_ids(file, index_dir):
	
	# Collect file ids in system_input.index.tab
	index_file_path = os.path.join(index_dir, "LC1-SimulatedMiniEvalP1.20220909.system_input.index.tab")
	index_df = pd.read_csv(index_file_path, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

	if index_df.shape[0] != 0:
		file_ids = []
		for index, row in index_df.iterrows():
			if row['file_id'] not in file_ids:
				file_ids.append(row['file_id'])
	
	# Check that file ids in file are same
	invalid_ids =[]
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	if df.shape[0] != 0:
		for index, row in df.iterrows():
			if row['file_id'] not in file_ids:
				invalid_ids.append(row['file_id'])
	
	if len(invalid_ids) > 0:
		logger.error("File {} contains file ids that are not found in file {}: {}".format(file, index_file_path, invalid_ids))
		return False
	
	return True

def check_file_data_types(file, header_types, file_type):
		
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	df_types = df.dtypes
	
	invalid_type_column = []
	if df.shape[0] != 0:
		for index, row in df.iterrows():
			for column in df.columns.values: # Validate that each value in each column is of the correct type
				if file_type != "valence": # Valence and arousal validation require particular validation checks
					if df_types[column] != header_types[column]:
							invalid_type_column.append(column)
				else: 
					valence_arousal_columns = ["valence_continuous", "valence_binned", "arousal_continuous", "arousal_binned"]
					if column not in valence_arousal_columns:
						if df_types[column] != header_types[column]:
								invalid_type_column.append(column)
					else: 
						if row[column] == "nospeech" and df_types[column] != 'object':
							invalid_type_column.append(column)
						elif row[column] != "nospeech" and is_float(row[column]) == False:
							invalid_type_column.append(column)
	
	if len(invalid_type_column) > 0:
		logger.error("Validation failed")
		logger.error("In file {}, the following column(s) have invalid data types: {}".format(file, invalid_type_column))
		return False
	
	return True

def check_fileid_segmentid_match(file, docs_dir):
    
	# Read in file ids and segment ids from segments.tab
	segments_file_path = os.path.join(docs_dir, "segments.tab")
	check_file_exist(segments_file_path, segments_file_path, docs_dir)
	segments_df = pd.read_csv(segments_file_path, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

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
				return False
			if len(invalid_segment_ids) > 0:
				logger.error("Segment ID(s) {} in {} is not found in {}".format(invalid_segment_ids, file, segments_file_path))
				return False
	return True

def check_duplicate_emotions(file):
		
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	for index, row in df.iterrows():
		emotion_list = [emotion.strip() for emotion in row['emotion'].split(',')] # Convert emotion column to list of individual emotions
		#emotion_list = row['emotion'].split(',')
		#uniq_emotions = set(emotion_list)
		if len(emotion_list) != len(set(emotion_list)):
			logger.error("Validation failed")
			seen = set()
			duplicates = [emotion for emotion in emotion_list if emotion in seen or seen.add(emotion)]
			if duplicates:
				logger.error("File {} contains the following emotion(s) duplicated: {}".format(file, duplicates))
			return False
	return True

def check_valid_emotions(file):
    	
	valid_emotions = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation", "none", "nospeech"]
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

	if df.shape[0] != 0:
		invalid_emotions = []
		for index, row in df.iterrows():
			#emotion_list = row['emotion'].split(', ')
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
	
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	
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
    	
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	valid_norms = ['001', '002', '007', 'none', 'nospeech'] #subject to change	
	
	if df.shape[0] != 0:
		invalid_norms = []
		for norm in df['norm']:
			if norm not in valid_norms:
				invalid_norms.append(norm)

		if len(invalid_norms) > 0:
			logger.error("Invalid file {}".format(file))
			logger.error("Additional emotion(s) '{}' have been found in {} ".format(set(invalid_norms), file))
			return False
	
	return True

def check_nospeech_all_columns(file):

	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

	if df.shape[0] != 0: 
		for index, row in df.iterrows():
			val_and_arousal_tags = [row['valence_continuous'], row['valence_binned'], row['arousal_continuous'], row['arousal_binned']]
			if "nospeech" in val_and_arousal_tags:
				nospeech_count = val_and_arousal_tags.count("nospeech")
				if nospeech_count != 4:
					logger.error("Validation failed. All columns should say 'nospeech'")
					logger.error("Expected 4 'nospeech' tags, only got {} in file {}".format(nospeech_count, file))
					return False
	return True
					
def check_nospeech_all_annotators(file):
	
	# Make sure "nospeech" tags applied to all value columns
	columns_valid = check_nospeech_all_columns(file)

	if columns_valid != True:
		return False
	
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	
	if df.shape[0] != 0: 
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
					logger.error("Invalid file: {}".format(file))
					logger.error("Inconsistent 'nospeech' tags in segment {}".format(current_segment_id))
					logger.error("The following annotators tagged 'nospeech': {}".format(set(nospeech_df['user_id'])))
					logger.error("The following annotators did not tag 'nospeech': {}".format(set(missing_nospeech_df['user_id'])))
					return False

			last_segment_id = current_segment_id
	return True

def check_valence_arousal_range(file):
    
	df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	
	valence_arousal_columns = ['valence_continuous', 'valence_binned', 'arousal_continuous', 'arousal_binned']
	
	if df.shape[0] != 0:
		invalid_ranges = []
		for index, row in df.iterrows():
			for column in valence_arousal_columns:
				if row[column] != "nospeech":
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
	file_info_df = pd.read_csv(file_info_path, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

	if file_info_df .shape[0] != 0:
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
	segments_df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
	
	if segments_df.shape[0] != 0:
		invalid_types = []
		for index, row in segments_df.iterrows():
			if row['file_id'] in text_ids:
				if row['start'].is_integer() == row['end'].is_integer() == False:
					invalid_types.append(row['start'])
					invalid_types.append(row['end'])
			elif row['file_id'] in audvid_ids:
				if isinstance(row['start'], float) == isinstance(row['end'], float) == False:
					invalid_types.append(row['start'])
					invalid_types.append(row['end'])
				if row['start'].is_integer() == row['end'].is_integer() == True:
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
