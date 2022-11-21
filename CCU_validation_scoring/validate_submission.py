import os
import sys
import logging
from .utils import *
from .validate import *

logger = logging.getLogger('VALIDATION')
logger.setLevel(logging.DEBUG)
format='%(levelname)s:%(name)s:%(message)s'
formatter = logging.Formatter(format)

h1 = logging.StreamHandler(sys.stdout)
h1.setLevel(logging.DEBUG)
h1.addFilter(lambda record: record.levelno <= logging.INFO)
h1.setFormatter(formatter)

h2 = logging.StreamHandler()
h2.setLevel(logging.WARNING)
h2.setFormatter(formatter)
logger.addHandler(h1)
logger.addHandler(h2)

def validate_nd_submission_dir_cli(args):

	#Please run validation of reference firstly to make sure reference is valid
	task = "norms"
	subm_file_dict = global_file_checks(args.reference_dir, args.submission_dir)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		type = subm_file_dict[subm_file]["type"]
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]
		
		if individual_file_check(task, type, subm_file_path, column_map, header_map, processed_label, subm_file, length, norm_list=None):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
		exit(1)
	else:
		logger.info('Validation succeeded')

def validate_ed_submission_dir_cli(args):

	#Please run validation of reference firstly to make sure reference is valid
	task = "emotions"
	subm_file_dict = global_file_checks(args.reference_dir, args.submission_dir)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		type = subm_file_dict[subm_file]["type"]
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]

		if individual_file_check(task, type, subm_file_path, column_map, header_map, processed_label, subm_file, length, norm_list=None):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
		exit(1)
	else:
		logger.info('Validation succeeded')

def validate_vd_submission_dir_cli(args):

	task = "valence_continuous"
	subm_file_dict = global_file_checks(args.reference_dir, args.submission_dir)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		type = subm_file_dict[subm_file]["type"]
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]

		if individual_file_check(task, type, subm_file_path, column_map, header_map, processed_label, subm_file, length, norm_list=None):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
		exit(1)
	else:
		logger.info('Validation succeeded')

def validate_ad_submission_dir_cli(args):

	task = "arousal_continuous"
	subm_file_dict = global_file_checks(args.reference_dir, args.submission_dir)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		type = subm_file_dict[subm_file]["type"]
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]

		if individual_file_check(task, type, subm_file_path, column_map, header_map, processed_label, subm_file, length, norm_list=None):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
		exit(1)
	else:
		logger.info('Validation succeeded')

def validate_cd_submission_dir_cli(args):

	task = "changepoint"
	subm_file_dict = global_file_checks(args.reference_dir, args.submission_dir)

	invalid_subm_file_path = []

	for subm_file in subm_file_dict:
		column_map, header_map = extract_modality_info(subm_file_dict[subm_file]["type"])
		type = subm_file_dict[subm_file]["type"]
		subm_file_path = subm_file_dict[subm_file]["path"]
		processed_label = subm_file_dict[subm_file]["processed"]
		length = subm_file_dict[subm_file]["length"]

		if individual_file_check(task, type, subm_file_path, column_map, header_map, processed_label, subm_file, length, norm_list=None):
			pass
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
		exit(1)
	else:
		logger.info('Validation succeeded')

def validate_ndmap_submission_dir_cli(args):

	mapping_file = os.path.join(args.submission_dir, "nd.map.tab")

	check_file_exist(mapping_file, mapping_file, args.submission_dir)

	column_map = {"ndmap": 3}
	header_map = {"ndmap":{"sys_norm": "object","ref_norm": "object","sub_id": "object"}}
	
	hidden_norm_list = load_list(args.hidden_norm_list_file)
	if individual_file_check("ndmap", None, mapping_file, column_map, header_map, processed_label=None, subm_file=None, length=None, norm_list=hidden_norm_list):
		logger.info('Validation succeeded')
	else:
		logger.error('Validation failed')
		exit(1)



