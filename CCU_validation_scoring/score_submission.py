import os
import pandas as pd
import logging
from .preprocess_reference import *
from .utils import *
from .score_norm_emotion import score_tad
from .score_valence_arousal import *
from .score_changepoint import *

logger = logging.getLogger('SCORING')

def score_nd_submission_dir_cli(args):

	try:
		scoring_index = pd.read_csv(args.scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:SCORING:{} is not a valid scoring index file'.format(args.scoring_index_file))
		exit(1)

	check_scoring_index_out_of_scope(args.reference_dir, scoring_index, "norms")

	if args.merge_ref_text_gap:
		merge_ref_text_gap = int(args.merge_ref_text_gap)
	else:
		merge_ref_text_gap = None

	if args.merge_ref_time_gap:
		merge_ref_time_gap = float(args.merge_ref_time_gap)
	else:
		merge_ref_time_gap = None

	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "norms", text_gap = merge_ref_text_gap, time_gap = merge_ref_time_gap)
	if args.norm_list_file:
		ref = process_subset_norm_emotion(args.norm_list_file, ref)
	hyp = preprocess_submission_file(args.submission_dir, args.reference_dir, scoring_index, "norms")

	if args.mapping_submission_dir:
		mapping_file = os.path.join(args.mapping_submission_dir, "nd.map.tab")
		mapping_df = pd.read_csv(mapping_file, dtype="object", sep = "\t")
	else:
		mapping_df = None

	if args.merge_sys_text_gap:
		merge_sys_text_gap = int(args.merge_sys_text_gap)
	else:
		merge_sys_text_gap = None

	if args.merge_sys_time_gap:
		merge_sys_time_gap = float(args.merge_sys_time_gap)
	else:
		merge_sys_time_gap = None

	merged_hyp = merge_sys_instance(hyp, merge_sys_text_gap, merge_sys_time_gap, args.combine_sys_llrs, args.merge_sys_label)

	thresholds = [float(i) for i in args.iou_thresholds.split(',')]
	score_tad(ref, merged_hyp, "norm", iou_thresholds=thresholds, output_dir=args.output_dir, mapping_df = mapping_df)

	print("Alignment")
	print("---------------")
	print(open(os.path.join(args.output_dir, 'instance_alignment.tab')).read())
	print("Class Scores")
	print("---------------")
	print(open(os.path.join(args.output_dir, 'scores_by_class.tab')).read())
	print("Aggregated Scores")
	print("-------------")
	print(open(os.path.join(args.output_dir, 'scores_aggregated.tab')).read())


def score_ed_submission_dir_cli(args):

	try:
		scoring_index = pd.read_csv(args.scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:SCORING:{} is not a valid scoring index file'.format(args.scoring_index_file))
		exit(1)

	check_scoring_index_out_of_scope(args.reference_dir, scoring_index, "emotions")
	
	if args.merge_ref_text_gap:
		merge_ref_text_gap = int(args.merge_ref_text_gap)
	else:
		merge_ref_text_gap = None

	if args.merge_ref_time_gap:
		merge_ref_time_gap = float(args.merge_ref_time_gap)
	else:
		merge_ref_time_gap = None

	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "emotions", text_gap = merge_ref_text_gap, time_gap = merge_ref_time_gap)
	if args.emotion_list_file:
		ref = process_subset_norm_emotion(args.emotion_list_file, ref)
	hyp = preprocess_submission_file(args.submission_dir, args.reference_dir, scoring_index, "emotions")

	if args.merge_sys_text_gap:
		merge_sys_text_gap = int(args.merge_sys_text_gap)
	else:
		merge_sys_text_gap = None

	if args.merge_sys_time_gap:
		merge_sys_time_gap = float(args.merge_sys_time_gap)
	else:
		merge_sys_time_gap = None

	merged_hyp = merge_sys_instance(hyp, merge_sys_text_gap, merge_sys_time_gap, args.combine_sys_llrs, args.merge_sys_label)

	thresholds = [float(i) for i in args.iou_thresholds.split(',')]
	score_tad(ref, merged_hyp, "emotion", iou_thresholds=thresholds, output_dir=args.output_dir, mapping_df = None)

	print("Alignment")
	print("---------------")
	print(open(os.path.join(args.output_dir, 'instance_alignment.tab')).read())
	print("Class Scores")
	print("---------------")
	print(open(os.path.join(args.output_dir, 'scores_by_class.tab')).read())
	print("Aggregated Scores")
	print("-------------")
	print(open(os.path.join(args.output_dir, 'scores_aggregated.tab')).read())


def score_vd_submission_dir_cli(args):

	try:
		scoring_index = pd.read_csv(args.scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:SCORING:{} is not a valid scoring index file'.format(args.scoring_index_file))
		exit(1)

	check_scoring_index_out_of_scope(args.reference_dir, scoring_index, "valence_continuous")
	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "valence_continuous")
	hyp = preprocess_submission_file(args.submission_dir, args.reference_dir, scoring_index, "valence_continuous")

	score_valence_arousal(ref, hyp, output_dir = args.output_dir, task = "valence_continuous")

	print("Diarization")
	print("---------------")
	print(open(os.path.join(args.output_dir, 'segment_diarization.tab')).read())
	print("Aggregated Scores")
	print("-------------")
	print(open(os.path.join(args.output_dir, 'scores_aggregated.tab')).read())
	

def score_ad_submission_dir_cli(args):

	try:
		scoring_index = pd.read_csv(args.scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:SCORING:{} is not a valid scoring index file'.format(args.scoring_index_file))
		exit(1)

	check_scoring_index_out_of_scope(args.reference_dir, scoring_index, "arousal_continuous")
	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "arousal_continuous")
	hyp = preprocess_submission_file(args.submission_dir, args.reference_dir, scoring_index, "arousal_continuous")

	score_valence_arousal(ref, hyp, output_dir = args.output_dir, task = "arousal_continuous")

	print("Diarization")
	print("---------------")
	print(open(os.path.join(args.output_dir, 'segment_diarization.tab')).read())
	print("Aggregated Scores")
	print("-------------")
	print(open(os.path.join(args.output_dir, 'scores_aggregated.tab')).read())

def score_cd_submission_dir_cli(args):

	try:
		scoring_index = pd.read_csv(args.scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:SCORING:{} is not a valid scoring index file'.format(args.scoring_index_file))
		exit(1)

	check_scoring_index_out_of_scope(args.reference_dir, scoring_index, "changepoint")
	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "changepoint")
	hyp = preprocess_submission_file(args.submission_dir, args.reference_dir, scoring_index, "changepoint")

	text_thresholds = [int(i) for i in args.delta_cp_text_thresholds.split(',')]
	time_thresholds = [float(i) for i in args.delta_cp_time_thresholds.split(',')]
	score_cp(ref, hyp, delta_cp_text_thresholds=text_thresholds, delta_cp_time_thresholds=time_thresholds, output_dir=args.output_dir)

	print("Alignment")
	print("---------------")
	print(open(os.path.join(args.output_dir, 'instance_alignment.tab')).read())
	print("Class Scores")
	print("---------------")
	print(open(os.path.join(args.output_dir, 'scores_by_class.tab')).read())







