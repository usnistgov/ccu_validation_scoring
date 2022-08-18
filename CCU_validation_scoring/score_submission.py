import os
import pandas as pd
import logging
import argparse
from .preprocess_reference import preprocess_reference_dir
from .utils import *
from .score_norm_emotion import score_tad
from .score_valence_arousal import *
from .score_changepoint import *

logger = logging.getLogger('SCORING')

def score_nd_submission_dir_cli(args):

	ref = preprocess_reference_dir(ref_dir = args.reference_dir, task = "norms")
	hyp = concatenate_submission_file(subm_dir = args.submission_dir, task = "norms")

	if args.mapping_submission_dir:
		hyp = mapping_known_hidden_norm(args.mapping_submission_dir, hyp)

	# print(ref)
	# print(hyp)
		
	thresholds = [float(i) for i in args.iou_thresholds.split(',')]
	score_tad(ref, hyp, "norm", iou_thresholds=thresholds, metrics=['map'], output_dir=args.output_dir, nb_jobs = -1)

	print("Class Scores")
	print("---------------")
	print(open(os.path.join(args.output_dir, 'class_scores.csv')).read())
	print("System Score")
	print("-------------")
	print(open(os.path.join(args.output_dir, 'system_scores.csv')).read())


def score_ed_submission_dir_cli(args):

	ref = preprocess_reference_dir(ref_dir = args.reference_dir, task = "emotions")
	hyp = concatenate_submission_file(subm_dir = args.submission_dir, task = "emotions")
	thresholds = [float(i) for i in args.iou_thresholds.split(',')]
	score_tad(ref, hyp, "emotion", iou_thresholds=thresholds, metrics=['map'], output_dir=args.output_dir, nb_jobs = -1)

	print("Class Scores")
	print("---------------")
	print(open(os.path.join(args.output_dir, 'class_scores.csv')).read())
	print("System Score")
	print("-------------")
	print(open(os.path.join(args.output_dir, 'system_scores.csv')).read())


def score_vd_submission_dir_cli(args):

	ref = preprocess_reference_dir(ref_dir = args.reference_dir, task = "valence_continuous")
	hyp = concatenate_submission_file(subm_dir = args.submission_dir, task = "valence_continuous")

	score_valence_arousal(ref, hyp, output_dir = args.output_dir, task = "valence_continuous")

	print("System Score")
	print("-------------")
	print(open(os.path.join(args.output_dir, 'system_scores.csv')).read())
	

def score_ad_submission_dir_cli(args):

	ref = preprocess_reference_dir(ref_dir = args.reference_dir, task = "arousal_continuous")
	hyp = concatenate_submission_file(subm_dir = args.submission_dir, task = "arousal_continuous")

	score_valence_arousal(ref, hyp, output_dir = args.output_dir, task = "arousal_continuous")

	print("System Score")
	print("-------------")
	print(open(os.path.join(args.output_dir, 'system_scores.csv')).read())

def score_cd_submission_dir_cli(args):

	ref = preprocess_reference_dir(ref_dir = args.reference_dir, task = "changepoint")
	hyp = concatenate_submission_file(subm_dir = args.submission_dir, task = "changepoint")
	text_thresholds = [int(i) for i in args.delta_cp_text_thresholds.split(',')]
	time_thresholds = [float(i) for i in args.delta_cp_time_thresholds.split(',')]
	score_cp(ref, hyp, delta_cp_text_thresholds=text_thresholds, delta_cp_time_thresholds=time_thresholds, output_dir=args.output_dir, nb_jobs = -1)

	print("System Score")
	print("-------------")
	print(open(os.path.join(args.output_dir, 'system_scores.csv')).read())







