import os
import pandas as pd
import logging
from .preprocess_reference import preprocess_reference_dir
from .validate import *

logger = logging.getLogger('VALIDATION')

def global_subm_checks(task, reference_dir, submission_dir):

	ref = preprocess_reference_dir(ref_dir = reference_dir, task = task)
	index_file_path, subm_file_dict = check_index_get_submission_files(ref, submission_dir)
	check_submission_files(submission_dir, index_file_path, subm_file_dict)

	return ref, subm_file_dict

def individual_subm_check(task, subm_file_path, column_map, header_map, processed_label, subm_file, length, ref_df):
	
	if task == "norms":
		subm_file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_start_small_end(subm_file_path) and
			check_start_end_within_length(subm_file_path, length))
	
	if task == "emotions":
		subm_file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_emotion_id(subm_file_path) and
			check_start_small_end(subm_file_path) and
			check_start_end_within_length(subm_file_path, length))

	if task == "valence_continuous" or task == "arousal_continuous":
		subm_file_checks = (check_valid_tab(subm_file_path) and
			check_column_number(subm_file_path,column_map[task]) and
			check_valid_header(subm_file_path,list(header_map[task])) and
			check_output_records(subm_file_path, processed_label) and
			check_data_type(subm_file_path, header_map[task]) and
			check_fileid_index_match(subm_file_path, subm_file) and
			check_start_small_end(subm_file_path) and
			check_time_no_gap(subm_file_path, header_map[task]) and
			check_duration_equal(subm_file_path, ref_df) and
			check_start_end_within_length(subm_file_path, length) and
			check_value_range(subm_file_path, task))
	
	return subm_file_checks

def validate_nd_submission_dir_cli(args):

	#Please run validation of reference firstly to make sure reference is valid
	task = "norms"
	ref, subm_file_dict = global_subm_checks(task, args.reference_dir, args.submission_dir)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]
		
		if individual_subm_check(task, subm_file_path, column_map, header_map, processed_label, subm_file, length, ref_df=None):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
	else:
		logger.info('Validation succedeed')


def validate_ed_submission_dir_cli(args):

	#Please run validation of reference firstly to make sure reference is valid
	task = "emotions"
	ref, subm_file_dict = global_subm_checks(task, args.reference_dir, args.submission_dir)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]

		if individual_subm_check(task, subm_file_path, column_map, header_map, processed_label, subm_file, length, ref_df=None):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
	else:
		logger.info('Validation succedeed')

def validate_vd_submission_dir_cli(args):

	task = "valence_continuous"
	ref, subm_file_dict = global_subm_checks(task, args.reference_dir, args.submission_dir)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]
		ref_df = ref.loc[ref["file_id"] == subm_file]

		if individual_subm_check(task, subm_file_path, column_map, header_map, processed_label, subm_file, length, ref_df=ref_df):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
	else:
		logger.info('Validation succedeed')

def validate_ad_submission_dir_cli(args):

	task = "arousal_continuous"
	ref, subm_file_dict = global_subm_checks(task, args.reference_dir, args.submission_dir)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]
		ref_df = ref.loc[ref["file_id"] == subm_file]

		if individual_subm_check(task, subm_file_path, column_map, header_map, processed_label, subm_file, length, ref_df=ref_df):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
	else:
		logger.info('Validation succedeed')

# def validate_cd_submission_dir_cli(args):

# 	index_df, subm_file_paths, ref_file_paths = get_reference_submission_files(ref_dir = args.reference_dir[0], 
# 																				subm_dir = args.submission_dir[0])

# 	invalid_subm_file_path = []

# 	for subm_file_path in subm_file_paths:
# 		doc_id, ref_file_path = find_corresponding_ref(subm_file_path,index_df,args.reference_dir[0],args.submission_dir[0])
# 		doc_type, column_map, header_map = extract_modality_info(ref_file_path)

# 		if (check_valid_tab(subm_file_path) and
# 			check_column_number(subm_file_path,column_map["CD"]) and
# 			check_valid_header(subm_file_path,list(header_map["CD"])) and
# 			check_data_type(subm_file_path, header_map["CD"]) and
# 			check_docid_index_match(subm_file_path, doc_id)):
# 			pass
# 		else:		
# 			invalid_subm_file_path.append(subm_file_path)		

# 	if len(invalid_subm_file_path) > 0:
# 		logger.error('Validation failed')
# 	else:
# 		logger.info('Validation succedeed')




