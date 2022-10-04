import os
import pandas as pd
import logging
import sys
from CCU_validation_scoring.validate import *

logger = logging.getLogger('VALIDATION')

def validate_ref_submission_dir_cli(args):
		
	ref_dir = args.reference_dir

	# Validate directory path
	is_directory = os.path.isdir(ref_dir)
	if is_directory != True:
		logger.error("Invalid directory: {} is not a directory".format(ref_dir))
		exit(1)

	# Iterate through directory & validate each file
	file_checks = False
	error_count = 0
	data_dir_files = []
	for root, dirs, files in os.walk(ref_dir, topdown=True):
		if root.endswith("data"):
			for file in files:
					
				# Retrieve file path
				file_path = os.path.join(root, file)
				# Retrieve paths to other index and docs directories
				docs_dir = os.path.join(root, '..',"docs")
				index_dir = os.path.join(root, '..',"index_files")
				error_count += 1 if file_checks != True else 0
				# Call file-specific checks
				if file == "valence_arousal.tab":
					file_checks = (global_ref_file_checks(file_path, index_dir) and 
					check_valence_arousal_range(file_path) and
					check_ref_fileid_segmentid_match(file_path, docs_dir) and
					check_noann_all_columns(file_path))
					error_count += 1 if file_checks != True else 0
				elif file == "emotions.tab": # CHANGE TO emotions.tab
					file_checks = (global_ref_file_checks(file_path, index_dir) and 
					check_empty_na(file_path, "emotions") and 
					check_duplicate_emotions(file_path) and 
					check_valid_emotions(file_path) and
					check_noann_all_columns(file_path) and 
					check_ref_fileid_segmentid_match(file_path, docs_dir))
					error_count += 1 if file_checks != True else 0
				elif file == "norms.tab":
					file_checks = (global_ref_file_checks(file_path, index_dir) and 
					check_empty_na(file_path, "norms") and 
					check_norm_range(file_path) and 
					check_ref_fileid_segmentid_match(file_path, docs_dir))
					error_count += 1 if file_checks != True else 0
				elif file == "changepoint.tab":
					file_checks = (global_ref_file_checks(file_path, index_dir))
					error_count += 1 if file_checks != True else 0
		elif root.endswith("docs"):
			for file in files:
				if file == "segments.tab":
					# Retrieve file path
					file_path = os.path.join(root, file)
					# Retrieve path to index directory
					index_dir = os.path.join(root, '..', "index_files")
					file_checks = (global_ref_file_checks(file_path, index_dir) and 
					check_start_end_types(file_path, root) and
					check_valid_timestamps(file_path, root))
					error_count += 1 if file_checks != True else 0

	if error_count > 0:
		logger.error("\nVALIDATION FAILED\n")
		logger.error("{} error(s) found in {} directory".format(error_count, ref_dir))
	else:
		print("\nVALIDATION SUCCEEDED\n")
	
	