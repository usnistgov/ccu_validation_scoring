import os
import pandas as pd
import logging
import argparse
from .check import *

logger = logging.getLogger('VALIDATION')

def validate_nd_submission_dir_cli(args):

	index_df, subm_file_paths, ref_file_paths = get_reference_submission_files(ref_dir = args.reference_dir[0], 
																				subm_dir = args.submission_dir[0])

	invalid_subm_file_path = []

	for subm_file_path in subm_file_paths:
		doc_id, ref_file_path = find_corresponding_ref(subm_file_path,index_df,args.reference_dir[0],args.submission_dir[0])
		doc_type, column_map, header_map = extract_modality_info(ref_file_path)

		if check_valid_csv(subm_file_path):
			if check_column_number(subm_file_path,column_map["ND"]):
				if check_valid_header(subm_file_path,list(header_map["ND"])):
					if check_data_type(subm_file_path, header_map["ND"]):
						if check_docid_index_match(subm_file_path, doc_id):
							if check_begin_small_end(subm_file_path):
								pass
							else:
								invalid_subm_file_path.append(subm_file_path)
						else:		
							invalid_subm_file_path.append(subm_file_path)		
					else:
						invalid_subm_file_path.append(subm_file_path)
				else:
					invalid_subm_file_path.append(subm_file_path)
			else:
				invalid_subm_file_path.append(subm_file_path)
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
	else:
		logger.info('Validation succedeed')


def validate_ed_submission_dir_cli(args):

	index_df, subm_file_paths, ref_file_paths = get_reference_submission_files(ref_dir = args.reference_dir[0], 
																				subm_dir = args.submission_dir[0])

	invalid_subm_file_path = []

	for subm_file_path in subm_file_paths:
		doc_id, ref_file_path = find_corresponding_ref(subm_file_path,index_df,args.reference_dir[0],args.submission_dir[0])
		doc_type, column_map, header_map = extract_modality_info(ref_file_path)

		if check_valid_csv(subm_file_path):
			if check_column_number(subm_file_path,column_map["ED"]):
				if check_valid_header(subm_file_path,list(header_map["ED"])):
					if check_data_type(subm_file_path, header_map["ED"]):
						if check_docid_index_match(subm_file_path, doc_id):
							if check_emotion_id(subm_file_path):
								if check_begin_small_end(subm_file_path):
									pass
								else:
									invalid_subm_file_path.append(subm_file_path)
							else:
								invalid_subm_file_path.append(subm_file_path)
						else:		
							invalid_subm_file_path.append(subm_file_path)		
					else:
						invalid_subm_file_path.append(subm_file_path)
				else:
					invalid_subm_file_path.append(subm_file_path)
			else:
				invalid_subm_file_path.append(subm_file_path)
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
	else:
		logger.info('Validation succedeed')

def validate_vd_submission_dir_cli(args):

	index_df, subm_file_paths, ref_file_paths = get_reference_submission_files(ref_dir = args.reference_dir[0], 
																				subm_dir = args.submission_dir[0])

	invalid_subm_file_path = []

	for subm_file_path in subm_file_paths:
		doc_id, ref_file_path = find_corresponding_ref(subm_file_path,index_df,args.reference_dir[0],args.submission_dir[0])
		doc_type, column_map, header_map = extract_modality_info(ref_file_path)

		if check_valid_csv(subm_file_path):
			if check_column_number(subm_file_path,column_map["VD"]):
				if check_valid_header(subm_file_path,list(header_map["VD"])):
					if check_data_type(subm_file_path, header_map["VD"]):
						if check_docid_index_match(subm_file_path, doc_id):
							if check_begin_small_end(subm_file_path):
								if check_time_no_gap(subm_file_path, header_map["VD"]):
									if check_duration_equal(subm_file_path, ref_file_path):
										pass
									else:
										invalid_subm_file_path.append(subm_file_path)
								else:
									invalid_subm_file_path.append(subm_file_path)
							else:
								invalid_subm_file_path.append(subm_file_path)
						else:
							invalid_subm_file_path.append(subm_file_path)
					else:		
						invalid_subm_file_path.append(subm_file_path)		
				else:
					invalid_subm_file_path.append(subm_file_path)
			else:
				invalid_subm_file_path.append(subm_file_path)
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
	else:
		logger.info('Validation succedeed')

def validate_ad_submission_dir_cli(args):

	index_df, subm_file_paths, ref_file_paths = get_reference_submission_files(ref_dir = args.reference_dir[0], 
																				subm_dir = args.submission_dir[0])

	invalid_subm_file_path = []

	for subm_file_path in subm_file_paths:
		doc_id, ref_file_path = find_corresponding_ref(subm_file_path,index_df,args.reference_dir[0],args.submission_dir[0])
		doc_type, column_map, header_map = extract_modality_info(ref_file_path)

		if check_valid_csv(subm_file_path):
			if check_column_number(subm_file_path,column_map["AD"]):
				if check_valid_header(subm_file_path,list(header_map["AD"])):
					if check_data_type(subm_file_path, header_map["AD"]):
						if check_docid_index_match(subm_file_path, doc_id):
							if check_begin_small_end(subm_file_path):
								if check_time_no_gap(subm_file_path, header_map["AD"]):
									if check_duration_equal(subm_file_path, ref_file_path):
										pass
									else:
										invalid_subm_file_path.append(subm_file_path)
								else:
									invalid_subm_file_path.append(subm_file_path)
							else:
								invalid_subm_file_path.append(subm_file_path)
						else:
							invalid_subm_file_path.append(subm_file_path)
					else:		
						invalid_subm_file_path.append(subm_file_path)		
				else:
					invalid_subm_file_path.append(subm_file_path)
			else:
				invalid_subm_file_path.append(subm_file_path)
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
	else:
		logger.info('Validation succedeed')

def validate_cd_submission_dir_cli(args):

	index_df, subm_file_paths, ref_file_paths = get_reference_submission_files(ref_dir = args.reference_dir[0], 
																				subm_dir = args.submission_dir[0])

	invalid_subm_file_path = []

	for subm_file_path in subm_file_paths:
		doc_id, ref_file_path = find_corresponding_ref(subm_file_path,index_df,args.reference_dir[0],args.submission_dir[0])
		doc_type, column_map, header_map = extract_modality_info(ref_file_path)

		if check_valid_csv(subm_file_path):
			if check_column_number(subm_file_path,column_map["CD"]):
				if check_valid_header(subm_file_path,list(header_map["CD"])):
					if check_data_type(subm_file_path, header_map["CD"]):
						if check_docid_index_match(subm_file_path, doc_id):
							pass
						else:		
							invalid_subm_file_path.append(subm_file_path)		
					else:
						invalid_subm_file_path.append(subm_file_path)
				else:
					invalid_subm_file_path.append(subm_file_path)
			else:
				invalid_subm_file_path.append(subm_file_path)
		else:
			invalid_subm_file_path.append(subm_file_path)

	if len(invalid_subm_file_path) > 0:
		logger.error('Validation failed')
	else:
		logger.info('Validation succedeed')




