import os
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger('VALIDATION')

def get_reference_submission_files(ref_dir, subm_dir):

	# Get and check index file
	index_file_path = os.path.join(subm_dir, "system_output.index.tab")
	if os.path.exists(index_file_path):
		pass
	else:
		logger.error('No index file found in {}'.format(index_file_path))
		logger.error('Validation failed')
		exit(1)

	# Build the reference files list
	# TODO: change all stuff related to reference dir since it will change to LDC structure
	ref_files = [ files for _, _, files in os.walk(ref_dir) ]

	if not ref_files:
		logger.error('No such directory {}'.format(ref_dir))
		logger.error('Validation failed')
		exit(1)

	ref_files = ref_files[0]

	ref_file_paths_list = [os.path.join(ref_dir, x) for x in ref_files]

	ref_docs = [os.path.splitext(x)[0] for x in ref_files]

	subm_files = [path.as_posix() for path in Path(subm_dir).rglob('*.tab')]

	index_df, subm_file_paths = check_index_file(index_file_path, ref_docs, subm_dir)

	index_subm_file_paths = subm_file_paths
	index_subm_file_paths.append(index_file_path)

	if set(subm_files) != set(index_subm_file_paths):
		logger.error('Invalid directory {}:'.format(subm_dir))

		# Check if we had not enough docs in the  reference files
		invalid_file = set(subm_files) - set(index_subm_file_paths)
		if invalid_file:
			logger.error("Additional file(s) {} have been found in submission {}".format(invalid_file, subm_dir))
 
		# Check whether we had too many docs in the reference files
		invalid_file = set(index_subm_file_paths) - set(subm_files)
		if invalid_file:
			logger.error("Additional file(s) {} have been found in submission {}".format(invalid_file, index_file_path))
			
		logger.error('Validation failed')
		exit(1)

	subm_file_paths.remove(index_file_path)

	return index_df, subm_file_paths, ref_file_paths_list

def check_valid_tab(subm_file_path):

	try:
		submission_df = pd.read_csv(subm_file_path, sep='\t')
		return True
	except Exception as e:
		logger.error('{} is not a valid tab file'.format(subm_file_path))
		return False

def check_column_number(subm_file_path, columns_number):

	submission_df = pd.read_csv(subm_file_path, sep='\t')

	if submission_df.shape[1] != columns_number:
		logger.error('Invalid file {}:'.format(subm_file_path))
		logger.error('File {} should contain {} columns.'.format(subm_file_path, columns_number))
		return False
	return True

def check_valid_header(subm_file_path, header):

	submission_df = pd.read_csv(subm_file_path, sep='\t')

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

def check_data_type(subm_file_path, header_type):

	submission_df = pd.read_csv(subm_file_path, sep='\t')
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
	invalid_emotion = []
	for i in submission_df["emotion_id"]:
		if i not in ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]:
			invalid_emotion.append(i)

	if len(invalid_emotion) > 0:
		logger.error('Invalid file {}:'.format(subm_file_path))
		logger.error("Additional emotion(s) '{}'' have been found in {}".format(set(invalid_emotion), subm_file_path))
		return False

def check_begin_small_end(subm_file_path):

	submission_df = pd.read_csv(subm_file_path, sep='\t')
	for i in range(submission_df.shape[0]):
		if submission_df.iloc[i]["begin"] >= submission_df.iloc[i]["end"]:
			logger.error('Invalid file {}:'.format(subm_file_path))
			logger.error("begin is equal to /larger than end in {}".format(subm_file_path))
			return False
	return True

def check_time_no_gap(subm_file_path, header_type):

	submission_df = pd.read_csv(subm_file_path, sep='\t')

	if header_type["begin"] == "int":
		for i in range(submission_df.shape[0]-1):
			if submission_df.iloc[i]["end"] + 1 != submission_df.iloc[i+1]["begin"]:
				logger.error('Invalid file {}:'.format(subm_file_path))
				logger.error("There are some gaps in timestamp of {}".format(subm_file_path))
				return False
		return True
	
	if header_type["begin"] == "float":
		for i in range(submission_df.shape[0]-1):
			if submission_df.iloc[i]["end"] != submission_df.iloc[i+1]["begin"]:
				logger.error('Invalid file {}:'.format(subm_file_path))
				logger.error("There are some gaps in timestamp of {}".format(subm_file_path))
				return False
		return True

def check_duration_equal(subm_file_path, ref_file_path):

	submission_df = pd.read_csv(subm_file_path, sep='\t')
	reference_df = pd.read_csv(ref_file_path, sep='\t')

	def calculate_duration(df):
		start = list(df["start"])
		end = list(df["end"])
		time_pool = start + end
		duration = max(time_pool)-min(time_pool)

		return duration

	if calculate_duration(submission_df) != calculate_duration(reference_df):
		logger.error('Invalid file {}:'.format(subm_file_path))
		logger.error("The duration of {} is different from the duration of {}".format(subm_file_path, ref_file_path))
		return False

	return True

def check_docid_index_match(subm_file_path, doc_id):

	submission_df = pd.read_csv(subm_file_path, sep='\t')
	for i in submission_df["doc_id"]:
		if i != doc_id:
			logger.error('Invalid file {}:'.format(subm_file_path))
			logger.error("doc_id in {} is different from doc_id in submission index file".format(subm_file_path))
			return False
	return True

def check_index_file(index_file_path, ref_docs, subm_dir):

	index_map = {"doc_id": "object","is_processed": "bool","message":"object","path":"object"}

	if (check_valid_tab(index_file_path) and
		check_column_number(index_file_path,4) and
		check_valid_header(index_file_path,list(index_map)) and
		check_data_type(index_file_path, index_map)):

		index_df = pd.read_csv(index_file_path, sep='\t')

		# Compare if reference doc is equal to index doc
		if set(list(index_df["doc_id"])) != set(ref_docs):
			logger.error('Doc_ids in reference dir are different from doc_ids in index_file')
			logger.error('Validation failed')
			exit(1)
		# Make sure is_processed true doc has corresponding path, otherwise no path
		na_paths = index_df["path"][index_df["is_processed"] == False]
		for i in na_paths:
			if i == i: # check if it's NaN value when is_processed false
				logger.error("Submission file paths of not processed file have been found")
				logger.error('Validation failed')
				exit(1)

		subm_file_paths = index_df["path"][index_df["is_processed"] == True]
		# Make sure each path is accessible
		subm_file_paths_list = []
		for j in subm_file_paths:
			if j != j:
				logger.error("Can't find submission file path of processed file")
				logger.error('Validation failed')
				exit(1)

			if j[:2] == './': #Check if path is start with ./
				j = j[2:]
			if subm_dir not in j: # Check it's absolute or relative path
				j = os.path.join(subm_dir, j)
			if os.path.exists(j):
				subm_file_paths_list.append(j)
			else:
				logger.error("Submission file path of processed file {} is inaccessible".format(j))
				logger.error('Validation failed')
				exit(1)
	else:
		logger.error('Validation failed')
		exit(1)

	return index_df, subm_file_paths_list

def find_corresponding_ref(subm_file_path,index_df,ref_dir,subm_dir):

	if subm_file_path in list(index_df["path"]):
		doc_id = index_df["doc_id"][index_df["path"] == subm_file_path] # If it's absolute path
	elif subm_file_path.replace(subm_dir+"/","") in list(index_df["path"]):
		doc_id = index_df["doc_id"][index_df["path"] == subm_file_path.replace(subm_dir+"/","")]
	else:							
		doc_id = index_df["doc_id"][index_df["path"] == "./" + subm_file_path.replace(subm_dir+"/","")] # If it's relative path

	ref_file_path = os.path.join(ref_dir, "{}.tab".format(doc_id.values[0]))
	if os.path.exists(ref_file_path):
		return doc_id.values[0], ref_file_path
	else:
		logger.error("Can't find corresponding reference of {}".format(subm_file_path))
		logger.error('Validation failed')
		exit(1)

def extract_modality_info(ref_file_path):

	reference_df = pd.read_csv(ref_file_path, sep='\t')
	doc_type = reference_df["modality"][0]
	if doc_type == "text":
		frame_data_type = "int"
	else:
		frame_data_type = "float"

	column_map = {"ND": 6, "ED": 5, "VD": 4, "AD": 4, "CD": 3}
	header_map = {"ND":{"doc_id": "object","norm_id": "int","begin": frame_data_type,"end": frame_data_type,"adherence": "bool","llr": "float"},
				"ED":{"doc_id": "object","emotion_id": "object","begin": frame_data_type,"end": frame_data_type,"llr": "float"},
				"VD":{"doc_id": "object","begin": frame_data_type,"end": frame_data_type,"valence": "int"},
				"AD":{"doc_id": "object","begin": frame_data_type,"end": frame_data_type,"arousal": "int"},
				"CD":{"doc_id": "object","point": frame_data_type,"llr": "float"}}

	return doc_type, column_map, header_map