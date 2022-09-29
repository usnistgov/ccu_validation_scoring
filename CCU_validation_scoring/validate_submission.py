import os
import pandas as pd
import logging
from .utils import *
from .validate import *

logger = logging.getLogger('VALIDATION')


def validate_nd_submission_dir_cli(args):

	#Please run validation of reference firstly to make sure reference is valid
	task = "norms"
	ref, subm_file_dict = global_file_checks(task, args.reference_dir, args.submission_dir, args.scoring_index_file)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		type = subm_file_dict[subm_file]["type"]
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]
		
		if individual_file_check(task, type, subm_file_path, column_map, header_map, processed_label, subm_file, length, ref_df=None, norm_list=None):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
		exit(1)
	else:
		logger.info('Validation succedeed')

def validate_ed_submission_dir_cli(args):

	#Please run validation of reference firstly to make sure reference is valid
	task = "emotions"
	ref, subm_file_dict = global_file_checks(task, args.reference_dir, args.submission_dir, args.scoring_index_file)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		type = subm_file_dict[subm_file]["type"]
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]

		if individual_file_check(task, type, subm_file_path, column_map, header_map, processed_label, subm_file, length, ref_df=None, norm_list=None):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
		exit(1)
	else:
		logger.info('Validation succedeed')

def validate_vd_submission_dir_cli(args):

	task = "valence_continuous"
	ref, subm_file_dict = global_file_checks(task, args.reference_dir, args.submission_dir, args.scoring_index_file)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		type = subm_file_dict[subm_file]["type"]
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]
		ref_df = ref.loc[ref["file_id"] == subm_file]

		if individual_file_check(task, type, subm_file_path, column_map, header_map, processed_label, subm_file, length, ref_df=ref_df, norm_list=None):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
		exit(1)
	else:
		logger.info('Validation succedeed')

def validate_ad_submission_dir_cli(args):

	task = "arousal_continuous"
	ref, subm_file_dict = global_file_checks(task, args.reference_dir, args.submission_dir, args.scoring_index_file)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		type = subm_file_dict[subm_file]["type"]
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]
		ref_df = ref.loc[ref["file_id"] == subm_file]

		if individual_file_check(task, type, subm_file_path, column_map, header_map, processed_label, subm_file, length, ref_df=ref_df, norm_list=None):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
		exit(1)
	else:
		logger.info('Validation succedeed')

def validate_cd_submission_dir_cli(args):

	task = "changepoint"
	ref, subm_file_dict = global_file_checks(task, args.reference_dir, args.submission_dir, args.scoring_index_file)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		type = subm_file_dict[subm_file]["type"]
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]

		if individual_file_check(task, type, subm_file_path, column_map, header_map, processed_label, subm_file, length, ref_df=None, norm_list=None):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
		exit(1)
	else:
		logger.info('Validation succedeed')

def validate_ndmap_submission_dir_cli(args):

	mapping_file = os.path.join(args.submission_dir, "nd.map.tab")

	check_file_exist(mapping_file, mapping_file, args.submission_dir)

	column_map = {"ndmap": 3}
	header_map = {"ndmap":{"sys_norm": "object","ref_norm": "object","sub_id": "object"}}
	
	hidden_norm_list = load_list(args.hidden_norm_list_file)
	if individual_file_check("ndmap", None, mapping_file, column_map, header_map, processed_label=None, subm_file=None, length=None, ref_df=None, norm_list=hidden_norm_list):
		logger.info('Validation succedeed')
	else:
		logger.error('Validation failed')
		exit(1)



