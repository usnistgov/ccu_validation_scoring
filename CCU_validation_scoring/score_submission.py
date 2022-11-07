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

	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "norms")
	if args.norm_list_file:
		ref = process_subset_norm_emotion(args.norm_list_file, ref)
	hyp = concatenate_submission_file(subm_dir = args.submission_dir, task = "norms")
	hyp_type = add_type_column(args.reference_dir, hyp)

	if args.mapping_submission_dir:
		mapping_file = os.path.join(args.mapping_submission_dir, "nd.map.tab")
		mapping_df = pd.read_csv(mapping_file, dtype="object", sep = "\t")
	else:
		mapping_df = None

	thresholds = [float(i) for i in args.iou_thresholds.split(',')]
	score_tad(ref, hyp_type, "norm", iou_thresholds=thresholds, output_dir=args.output_dir, mapping_df = mapping_df)

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

	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "emotions")
	if args.emotion_list_file:
		ref = process_subset_norm_emotion(args.emotion_list_file, ref)
	hyp = concatenate_submission_file(subm_dir = args.submission_dir, task = "emotions")
	hyp_type = add_type_column(args.reference_dir, hyp)

	thresholds = [float(i) for i in args.iou_thresholds.split(',')]
	score_tad(ref, hyp_type, "emotion", iou_thresholds=thresholds, output_dir=args.output_dir, mapping_df = None)

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

	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "valence_continuous")
	hyp = concatenate_submission_file(subm_dir = args.submission_dir, task = "valence_continuous")
	hyp_type = add_type_column(args.reference_dir, hyp)

	score_valence_arousal(ref, hyp_type, output_dir = args.output_dir, task = "valence_continuous")

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

	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "arousal_continuous")
	hyp = concatenate_submission_file(subm_dir = args.submission_dir, task = "arousal_continuous")
	hyp_type = add_type_column(args.reference_dir, hyp)

	score_valence_arousal(ref, hyp_type, output_dir = args.output_dir, task = "arousal_continuous")

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

	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "changepoint")
	hyp = concatenate_submission_file(subm_dir = args.submission_dir, task = "changepoint")
	hyp_type = add_type_column(args.reference_dir, hyp)

	text_thresholds = [int(i) for i in args.delta_cp_text_thresholds.split(',')]
	time_thresholds = [float(i) for i in args.delta_cp_time_thresholds.split(',')]
	score_cp(ref, hyp_type, delta_cp_text_thresholds=text_thresholds, delta_cp_time_thresholds=time_thresholds, output_dir=args.output_dir)

	print("Alignment")
	print("---------------")
	print(open(os.path.join(args.output_dir, 'instance_alignment.tab')).read())
	print("Class Scores")
	print("---------------")
	print(open(os.path.join(args.output_dir, 'scores_by_class.tab')).read())







