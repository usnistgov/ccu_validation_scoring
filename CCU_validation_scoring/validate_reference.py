# python3 validate_reference.py <directory type = data or docs> <directory path> 
# For example:
# python3 validate_reference.py data /Users/cnc30/Documents/VisualStudioProjects/ccu_validation_scoring/test/reference/LDC_reference_sample/data /Users/cnc30/Documents/VisualStudioProjects/ccu_validation_scoring/test/reference/LDC_reference_sample/index_files /Users/cnc30/Documents/VisualStudioProjects/ccu_validation_scoring/test/reference/LDC_reference_sample/docs
# python3 validate_reference.py docs /Users/cnc30/Documents/VisualStudioProjects/ccu_validation_scoring/test/reference/LDC_reference_sample/docs /Users/cnc30/Documents/VisualStudioProjects/ccu_validation_scoring/test/reference/LDC_reference_sample/index_files
import os
import pandas as pd
import logging
import sys
from CCU_validation_scoring.validate import *
from CCU_validation_scoring.reference_validation import *

logger = logging.getLogger('VALIDATION')

def validate_data_ref_dir_cli(args):
	
	# Store directory type (data, docs, or index), directory path, & the path to the index file directory
	if args[0] == "docs" and len(args) == 3:
		dir_type = args[0]
		dir_path = args[1]
		index_dir = args[2]
	elif args[0] == "data" and len(args) == 4:
		dir_type = args[0]
		dir_path = args[1]
		index_dir = args[2]
		docs_dir = args[3]
	else: 
		logger.error("Invalid arguments")
		logger.error("Command to validate data directory: python3 validate_reference.py data <path to data directory> <path to index directory> <path to docs directory>")
		logger.error("Command to validate docs directory: python3 validate_reference.py docs <path to docs directory> <path to index directory>")

	# Validate directory path
	is_directory = os.path.isdir(dir_path)
	if is_directory != True:
		logger.error("Invalid directory: {} is not a directory".format(dir_path))
		exit(1)
	
	# Validates changepoint.tab, emotions.tab, norms.tab, valence_arousal.tab, and segments.tab
	file_checks = False
	error_count = 0
	file_list = os.listdir(dir_path)
	for file in file_list:
		full_file_path = os.path.join(dir_path, file)

		if dir_type == "data":
			# Call global checks
			file_checks = global_checks(full_file_path, index_dir)
			# Call file-specific checks
			if file == "valence_arousal.tab":
				file_checks = (check_nospeech_all_annotators(full_file_path) and 
				check_valence_arousal_range(full_file_path) and
				check_fileid_segmentid_match(full_file_path, docs_dir))
				error_count += 1 if file_checks != True else 0
			if file == "emotions.tab":
				file_checks = (check_empty_na(full_file_path, "emotions") and 
				check_duplicate_emotions(full_file_path) and 
				check_valid_emotions(full_file_path) and
				check_fileid_segmentid_match(full_file_path, docs_dir))
				error_count += 1 if file_checks != True else 0
			if file == "norms.tab":
				file_checks = (check_empty_na(full_file_path, "norms") and 
				check_norm_range(full_file_path) and 
				check_fileid_segmentid_match(full_file_path, docs_dir))
				error_count += 1 if file_checks != True else 0
		elif dir_type == "docs":
			if file == 'segments.tab':
				file_checks = (global_checks(full_file_path, index_dir) and 
				check_start_small_end(full_file_path) and
				check_start_end_types(full_file_path, dir_path))
				error_count += 1 if file_checks != True else 0
	
	if error_count > 0:
		logger.error("{} error(s) found in {} directory".format(error_count, dir_path))
		return False
	else:
		return True

if __name__ == "__main__":

	file_checks = validate_data_ref_dir_cli(sys.argv[1:])

	if file_checks != True:
		logger.error("\nVALIDATION FAILED\n")
	else: 
		print("\nVALIDATION SUCCEEDED\n")
	