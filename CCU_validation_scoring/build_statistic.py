import os,glob
import numpy as np
import pandas as pd

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

def class_type_instance_count(df, data):

	class_type_instance_count = df.groupby(['Class', 'type']).size().reset_index(name='value')
	class_type_instance_count.rename(columns = {"Class": "class", "type": "genre"}, inplace = True)
	class_type_instance_count["metric"] = "instance_counts"
	class_type_instance_count["data"] = data

	class_type_instance_count = class_type_instance_count[["data","class","genre","metric","value"]]

	return class_type_instance_count

def type_instance_count(df, data):

	type_instance_count = df.groupby(['type']).size().reset_index(name='value')
	type_instance_count.rename(columns = {"type": "genre"}, inplace = True)
	type_instance_count["class"] = "cp"
	type_instance_count["metric"] = "instance_counts"
	type_instance_count["data"] = data

	type_instance_count = type_instance_count[["data","class","genre","metric","value"]]

	return type_instance_count

def type_time_sum(df, data):

	type_time_sum = df.groupby(['type'])['total_seconds/characters'].sum().reset_index()
	type_time_sum['total_seconds/characters'] = type_time_sum['total_seconds/characters'].apply(formatNumber)
	type_time_sum.rename(columns = {"type": "genre"}, inplace = True)
	type_time_sum["data"] = data
	type_time_sum.rename(columns = {"total_seconds/characters": "value"}, inplace = True)
	type_time_sum['metric'] = np.where(type_time_sum['genre'] == 'text', "total_instance_characters", "total_instance_seconds")

	type_time_sum = type_time_sum[["data","genre","metric","value"]]

	return type_time_sum

def type_class_avg(df, data):

	type_class_avg = df.groupby(['type'])['Class'].mean().reset_index()
	type_class_avg.rename(columns={'Class': "value"}, inplace=True)
	type_class_avg["data"] = data
	type_class_avg["metric"] = "mean_level"
	type_class_avg.rename(columns = {"type": "genre"}, inplace = True)

	type_class_avg = type_class_avg[["data","genre","metric","value"]]

	return type_class_avg

def type_class_std(df, data):

	type_class_std = df.groupby(['type'])['Class'].std().reset_index()
	type_class_std.rename(columns={'Class': "value"}, inplace=True)
	type_class_std["data"] = data
	type_class_std["metric"] = "stdev_level"
	type_class_std.rename(columns = {"type": "genre"}, inplace = True)

	type_class_std = type_class_std[["data","genre","metric","value"]]

	return type_class_std

def unique_file_count(df):

	file_id = df.file_id.unique()

	return len(file_id)

def combine_result(*arguments):

	final = pd.concat(arguments, ignore_index=True)

	return final

def generate_agg_row(value, data, label):

	df = pd.DataFrame({"data": data, "genre": "all", "metric": label, "value": value}, index=[0])

	return df

def statistic(reference_dir, ref, submission_dir, hyp, output_dir, task):

	system_input_index_file_path = glob.glob(os.path.join(reference_dir, "index_files", "*system_input.index.tab"))[0]
	system_input_index_df = pd.read_csv(system_input_index_file_path, sep='\t')
	#file_count: ref file counts in system input index
	ref_unique_file_count = generate_agg_row(unique_file_count(system_input_index_df), "ref", "file_counts")

	system_output_index_file_path = os.path.join(submission_dir, "system_output.index.tab")
	system_output_index_df = pd.read_csv(system_output_index_file_path, dtype={'message': object}, sep='\t')
	#file_count: sys file counts in system output index
	sys_unique_file_count = generate_agg_row(unique_file_count(system_output_index_df), "sys", "file_counts")

	ref_ann_prune = ref[(ref['Class'] != "noann")]
	#file_scoring_counts: file counts in ref/sys that ready to score for specific task
	ref_unique_file_count_ann = generate_agg_row(unique_file_count(ref_ann_prune), "ref", "file_scoring_counts")
	sys_unique_file_count_ann = generate_agg_row(unique_file_count(hyp), "sys", "file_scoring_counts")
	ref_noann_prune = ref[(ref['Class'] == "noann")]
	#noann_segment_scoring_counts: “noann” segment counts in ref that ready to score for specific task
	ref_unique_file_count_noann = generate_agg_row(len(ref_noann_prune), "ref", "noann_segment_scoring_counts")

	if task == "norms" or task == "emotions":

		ref_class_type_instance_count = class_type_instance_count(ref_ann_prune, "ref")
		sys_class_type_instance_count = class_type_instance_count(hyp, "sys")

		statistic_class = combine_result(ref_class_type_instance_count, sys_class_type_instance_count)

		statistic_class.to_csv(os.path.join(output_dir, "statistics_by_class.tab"), sep = "\t", index = None)

		ref_ann_prune_copy = ref_ann_prune.copy()
		ref_ann_prune_copy['total_seconds/characters'] = ref_ann_prune_copy.apply(generate_diff, axis=1)
		ref_type_time_sum = type_time_sum(ref_ann_prune_copy, "ref")

		if hyp.shape[0] > 0:
			hyp_copy = hyp.copy()
			hyp_copy['total_seconds/characters'] = hyp_copy.apply(generate_diff, axis=1)
			sys_type_time_sum = type_time_sum(hyp_copy, "sys")
			statistic_aggregated = combine_result(ref_type_time_sum, sys_type_time_sum, ref_unique_file_count, ref_unique_file_count_ann, ref_unique_file_count_noann, sys_unique_file_count, sys_unique_file_count_ann)
		else:
			statistic_aggregated = combine_result(ref_type_time_sum, ref_unique_file_count, ref_unique_file_count_ann, ref_unique_file_count_noann, sys_unique_file_count, sys_unique_file_count_ann)

		statistic_aggregated.to_csv(os.path.join(output_dir, "statistics_aggregated.tab"), sep = "\t", index = None)

	if task == "valence_continuous" or task == "arousal_continuous":
		ref_type_class_avg = type_class_avg(ref_ann_prune, "ref")
		ref_type_class_std = type_class_std(ref_ann_prune, "ref")

		sys_type_class_avg = type_class_avg(hyp, "sys")
		sys_type_class_std = type_class_std(hyp, "sys")

		statistic_aggregated = combine_result(ref_type_class_avg, ref_type_class_std, sys_type_class_avg, sys_type_class_std, ref_unique_file_count, ref_unique_file_count_ann, ref_unique_file_count_noann, sys_unique_file_count, sys_unique_file_count_ann)
		statistic_aggregated["value"] = statistic_aggregated["value"].apply(formatNumber)

		statistic_aggregated.to_csv(os.path.join(output_dir, "statistics_aggregated.tab"), sep = "\t", index = None)

	if task == "changepoint":
		ref_type_instance_count = type_instance_count(ref_ann_prune, "ref")
		sys_type_instance_count = type_instance_count(hyp, "sys")

		statistic_class = combine_result(ref_type_instance_count, sys_type_instance_count)

		statistic_class.to_csv(os.path.join(output_dir, "statistics_by_class.tab"), sep = "\t", index = None)

		statistic_aggregated = combine_result(ref_unique_file_count, ref_unique_file_count_ann, ref_unique_file_count_noann, sys_unique_file_count, sys_unique_file_count_ann)

		statistic_aggregated.to_csv(os.path.join(output_dir, "statistics_aggregated.tab"), sep = "\t", index = None)

