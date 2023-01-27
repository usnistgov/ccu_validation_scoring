import os,glob
import pandas as pd

global_known_norm_list = ["101","102","103","104","105"]
global_hidden_norm_list = ["106","107"]

def generate_norm_emotion_list(task, unique_class_list):

	if task == "norms":
		known_norm_list = [x for x in unique_class_list if x in global_known_norm_list]
		hidden_norm_list = [x for x in unique_class_list if x in global_hidden_norm_list]		
		other_norm_list = [x for x in unique_class_list if x not in global_known_norm_list and x not in global_hidden_norm_list]
		print("The unique known {} list: {}".format(task, known_norm_list))
		if hidden_norm_list != []:
			print("The unique hidden {} list: {}".format(task, hidden_norm_list))
		if other_norm_list != []:
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
	class_instance_count = df.groupby(['Class']).size().reset_index(name='segment_counts')
	return class_instance_count

def class_type_instance_count(df):
	class_type_instance_count = df.groupby(['Class', 'type']).size().reset_index(name='segment_counts')
	return class_type_instance_count

def class_type_time_sum(df):
	class_type_time_sum = df.groupby(['Class', 'type'])['total_seconds/characters'].sum().reset_index()
	class_type_time_sum['total_seconds/characters'] = class_type_time_sum['total_seconds/characters'].apply(formatNumber)
	return class_type_time_sum

def type_instance_count(df):
	type_instance_count = df.groupby(['type']).size().reset_index(name='segment_counts')
	return type_instance_count

def type_time_sum(df):
	type_time_sum = df.groupby(['type'])['total_seconds/characters'].sum().reset_index()
	type_time_sum['total_seconds/characters'] = type_time_sum['total_seconds/characters'].apply(formatNumber)
	return type_time_sum

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

def reference_statistic(reference_dir, ref, task):

	system_input_index_file_path = os.path.join(reference_dir, "index_files", "*system_input.index.tab")
	system_input_index_file_path = glob.glob(system_input_index_file_path)[0]
	system_input_index_df = pd.read_csv(system_input_index_file_path, sep='\t')
	unique_file_count_result = unique_file_count(system_input_index_df)
	print("The number of unique files in reference: {}".format(unique_file_count_result))
	unique_file_count_annotation_result = unique_file_count(ref)
	print("The number of unique files in reference for {} scoring: {}".format(task, unique_file_count_annotation_result))
	ref_noann_prune = ref[(ref['Class'] != "noann")]
	print("The number of segments in reference for {} scoring: {}".format(task, len(ref)))
	print("The number of noann in reference for {} scoring: {}".format(task, len(ref)-len(ref_noann_prune)))
	type_instance_count_result = type_instance_count(ref_noann_prune)
	print(type_instance_count_result)

	if task == "norms" or task == "emotions":
		unique_class_result = unique_class(ref_noann_prune)
		generate_norm_emotion_list(task, unique_class_result)
		class_instance_count_result = class_instance_count(ref_noann_prune)
		class_type_instance_count_result = class_type_instance_count(ref_noann_prune)
		print(class_instance_count_result)
		print(class_type_instance_count_result)

		ref_noann_prune_copy = ref_noann_prune.copy()
		ref_noann_prune_copy['total_seconds/characters'] = ref_noann_prune_copy.apply(generate_diff, axis=1)
		class_type_time_sum_result = class_type_time_sum(ref_noann_prune_copy)
		type_time_sum_result = type_time_sum(ref_noann_prune_copy)
		print(class_type_time_sum_result)
		print(type_time_sum_result)

	if task == "valence_continuous" or task == "arousal_continuous":
		type_class_avg_result = type_class_avg(ref_noann_prune, task)
		type_class_median_result = type_class_median(ref_noann_prune, task)
		print(type_class_avg_result)
		print(type_class_median_result)

		ref_noann_prune_copy = ref_noann_prune.copy()
		ref_noann_prune_copy['total_seconds/characters'] = ref_noann_prune_copy.apply(generate_diff, axis=1)
		type_time_sum_result = type_time_sum(ref_noann_prune_copy)
		print(type_time_sum_result)

def submission_statistic(submission_dir, hyp, task):

	index_file_path = os.path.join(submission_dir, "system_output.index.tab")
	index_df = pd.read_csv(index_file_path, dtype={'message': object}, sep='\t')
	unique_file_count_result = unique_file_count(index_df)
	print("The number of unique files in submission: {}".format(unique_file_count_result))
	unique_file_count_annotation_result = unique_file_count(hyp)
	print("The number of unique files in submission for {} scoring: {}".format(task, unique_file_count_annotation_result))
	print("The number of segments in submission for {} scoring: {}".format(task, len(hyp)))
	type_instance_count_result = type_instance_count(hyp)
	print(type_instance_count_result)

	if task == "norms" or task == "emotions":
		unique_class_result = unique_class(hyp)
		generate_norm_emotion_list(task, unique_class_result)
		class_instance_count_result = class_instance_count(hyp)
		class_type_instance_count_result = class_type_instance_count(hyp)
		print(class_instance_count_result)
		print(class_type_instance_count_result)

		hyp_copy = hyp.copy()
		hyp_copy['total_seconds/characters'] = hyp_copy.apply(generate_diff, axis=1)
		class_type_time_sum_result = class_type_time_sum(hyp_copy)
		type_time_sum_result = type_time_sum(hyp_copy)
		print(class_type_time_sum_result)
		print(type_time_sum_result)		

	if task == "valence_continuous" or task == "arousal_continuous":
		type_class_avg_result = type_class_avg(hyp, task)
		type_class_median_result = type_class_median(hyp, task)
		print(type_class_avg_result)
		print(type_class_median_result)

		hyp_copy = hyp.copy()
		hyp_copy['total_seconds/characters'] = hyp_copy.apply(generate_diff, axis=1)
		type_time_sum_result = type_time_sum(hyp_copy)
		print(type_time_sum_result)
