import os
import argparse
import pandas as pd
from CCU_validation_scoring.preprocess_reference import *

global_known_norm_list = ["101","102","103","104","105"]

def generate_norm_emotion_list(task, unique_class_list):

	if task == "norms":
		known_norm_list = [x for x in unique_class_list if x in global_known_norm_list]
		other_norm_list = [x for x in unique_class_list if x not in global_known_norm_list]
		print("The unique known {} list: {}".format(task, known_norm_list))
		print("The unique other {} list: {}".format(task, other_norm_list))
	else:
		print("The unique {} list: {}".format(task, unique_class_list))

def formatNumber(num):
	"""
	Convert integer number into integer format
	"""
	if num % 1 == 0:
		return str(int(num))
	else:
		return str(round(num,2))

def generate_diff(row):

	if row['type'] == "text":
		val = row['end'] - row['start'] + 1
	else:
		val = row['end'] - row['start']

	return val

def class_instance_count(df):
	class_instance_count = df.groupby(['Class']).size().reset_index(name='instance_counts')
	return class_instance_count

def class_type_instance_count(df):
	class_type_instance_count = df.groupby(['Class', 'type']).size().reset_index(name='instance_counts')
	return class_type_instance_count

def class_type_time_sum(df):
	df['total_seconds/characters'] = df.apply(generate_diff, axis=1)
	class_type_time_sum = df.groupby(['Class', 'type'])['total_seconds/characters'].sum().reset_index()
	class_type_time_sum['total_seconds/characters'] = class_type_time_sum['total_seconds/characters'].apply(formatNumber)
	return class_type_time_sum

def type_class_avg(df, task):
	type_class_avg = df.groupby(['type'])['Class'].mean().reset_index()
	type_class_avg.rename(columns={'Class':'average_{}'.format(task)}, inplace=True)
	return type_class_avg

def type_class_median(df, task):
	type_class_median = df.groupby(['type'])['Class'].median().reset_index()
	type_class_median.rename(columns={'Class':'median_{}'.format(task)}, inplace=True)
	return type_class_median

def unique_file_count(df):
	file_id = df.file_id.unique()
	return len(file_id)

def unique_class(df):
	Class = sorted(list(df.Class.unique()))
	return Class

def reference_statistic(reference_dir, scoring_index, task):

	ref = preprocess_reference_dir(reference_dir, scoring_index, task)
	ref_noann_prune = ref[(ref['Class'] != "noann")]
	unique_file_count_result = unique_file_count(ref_noann_prune)
	print("The number of unique files in reference: {}".format(unique_file_count_result))

	if task == "norms" or task == "emotions":
		unique_class_result = unique_class(ref_noann_prune)
		generate_norm_emotion_list(task, unique_class_result)
		class_instance_count_result = class_instance_count(ref_noann_prune)
		class_type_instance_count_result = class_type_instance_count(ref_noann_prune)
		print(class_instance_count_result)
		print(class_type_instance_count_result)

		ref_noann_prune_copy = ref_noann_prune.copy()
		class_type_time_sum_result = class_type_time_sum(ref_noann_prune_copy)
		print(class_type_time_sum_result)

	if task == "valence_continuous" or task == "arousal_continuous":
		type_class_avg_result = type_class_avg(ref_noann_prune, task)
		type_class_median_result = type_class_median(ref_noann_prune, task)
		print(type_class_avg_result)
		print(type_class_median_result)

	if task == "changepoint":
		print(ref_noann_prune)

def submission_statistic(submission_dir, reference_dir, scoring_index, task):

	hyp = preprocess_submission_file(submission_dir, reference_dir, scoring_index, task)
	hyp_noann_prune = hyp[(hyp['Class'] != "noann")]
	unique_file_count_result = unique_file_count(hyp_noann_prune)
	print("The number of unique files in submission: {}".format(unique_file_count_result))

	if task == "norms" or task == "emotions":
		unique_class_result = unique_class(hyp_noann_prune)
		generate_norm_emotion_list(task, unique_class_result)
		class_instance_count_result = class_instance_count(hyp_noann_prune)
		class_type_instance_count_result = class_type_instance_count(hyp_noann_prune)
		print(class_instance_count_result)
		print(class_type_instance_count_result)

		hyp_noann_prune_copy = hyp_noann_prune.copy()
		class_type_time_sum_result = class_type_time_sum(hyp_noann_prune_copy)
		print(class_type_time_sum_result)

	if task == "valence_continuous" or task == "arousal_continuous":
		type_class_avg_result = type_class_avg(hyp_noann_prune, task)
		type_class_median_result = type_class_median(hyp_noann_prune, task)
		print(type_class_avg_result)
		print(type_class_median_result)

	if task == "changepoint":
		print(hyp_noann_prune)

def main():

	parser = argparse.ArgumentParser(description='generate statistic for reference or submission')
	parser.add_argument('-r','--reference-dir', type=str, required=True, help='reference-dir')
	parser.add_argument('-s','--submission-dir', type=str, help='submission-dir')
	parser.add_argument('-i','--scoring-index-file', type=str, required=True, help='Use to filter file from scoring')
	parser.add_argument('-t','--task', choices=['norms', 'emotions', 'valence_continuous', 'arousal_continuous', 'changepoint'], required=True, help = 'norms, emotions, valence_continuous, arousal_continuous, changepoint')
	args = parser.parse_args()

	scoring_index = pd.read_csv(args.scoring_index_file, usecols = ['file_id'], sep = "\t")
	if not args.submission_dir:
		reference_statistic(args.reference_dir, scoring_index, args.task)
	if args.submission_dir and args.reference_dir:
		submission_statistic(args.submission_dir, args.reference_dir, scoring_index, args.task)	

if __name__ == '__main__':
	main()