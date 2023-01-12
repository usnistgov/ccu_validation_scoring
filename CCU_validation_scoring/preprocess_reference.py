import os
from time import time
from unittest import result
import pandas as pd
from .utils import *

silence_string = "noann"

def process_subset_norm_emotion(list_file, ref):
	""" 
	Method to check filter args and apply them to ref so they can be used by
	multiple commands.    
	"""
	norm_emotion_list = load_list(list_file)
	# Exclude all classes but include relevant norm in reference
	pruned_ref = ref.loc[(ref.Class.isin(norm_emotion_list)) | (ref.Class == silence_string)]
	
	return pruned_ref

def extend_gap_segment(ref):
	"""
	Extend the gap between end of previous segment and start of next segment in reference.

	e.g.
	From
	Seg  Start  End
	Seg1 45.2   60.2
	Seg2 60.201 75.201
	To result
	Seg  Start  End
	Seg1 45.2   60.201
	Seg2 60.201   75.201

	Parameters
	----------
	ref

	Returns
	-------
	ref_sorted: data frame after extending the gap

	"""

	gap_map = {"text": 10, "audio": 1, "video": 1}
	class_type = list(ref["Class_type"])[0]
	for i in range(1,ref.shape[0]):
		if ref.iloc[i]["file_id"] == ref.iloc[i-1]["file_id"]:
			diff = round(ref.iloc[i]["start"] - ref.iloc[i-1]["end"],3)
			gap = gap_map[ref.iloc[i]["type"]]

			if ref.iloc[i]["type"] == "text":
				if diff < gap:
					ref.iloc[i-1, ref.columns.get_loc('end')] = ref.iloc[i]["start"] - 1
				else:
					ref.loc[len(ref.index)] = [ref.iloc[i]["file_id"], silence_string, ref.iloc[i-1, ref.columns.get_loc('end')] + 1, ref.iloc[i]["start"] - 1, class_type, ref.iloc[i]["type"], ref.iloc[i]["length"]]
			elif ref.iloc[i]["type"] in ["audio", "video"]:
				if diff < gap:
					ref.iloc[i-1, ref.columns.get_loc('end')] = ref.iloc[i]["start"]
				else:
					ref.loc[len(ref.index)] = [ref.iloc[i]["file_id"], silence_string, ref.iloc[i-1, ref.columns.get_loc('end')], ref.iloc[i]["start"], class_type, ref.iloc[i]["type"], ref.iloc[i]["length"]]

	ref_sorted = ref.sort_values(by=["file_id","start","end"]).reset_index(drop=True)
	return ref_sorted

def fill_start_end(ref):
	"""
	Add no score region to start and end of the file in reference.

	e.g.
	From
	Start  End  Class
	45.2   60.2  joy

	To result
	Start  End  Class
	0  		45.2	noann
	45.2   60.2  joy

	Parameters
	----------
	ref

	Returns
	-------
	ref_sorted: data frame after adding the no score region

	"""
	
	class_type = list(ref["Class_type"])[0]
	file_ids = get_unique_items_in_array(ref['file_id'])
	for i in file_ids:
		sub_ref = extract_df(ref, i)
		type = list(sub_ref["type"])[0]
		length = list(sub_ref["length"])[0]
		start = min(list(sub_ref["start"]))
		end = max(list(sub_ref["end"]))

		if type == "text":
			if start > 0:
				ref.loc[len(ref.index)] = [i, silence_string, 0, start - 1, class_type, type, length]
			if end < length:
				ref.loc[len(ref.index)] = [i, silence_string, end + 1, length, class_type, type, length]

		if type in ["audio","video"]:
			if start > 0:
				ref.loc[len(ref.index)] = [i, silence_string, 0, start, class_type, type, length]
			if end < length:
				ref.loc[len(ref.index)] = [i, silence_string, end, length, class_type, type, length]
	
	ref_sorted = ref.sort_values(by=["file_id","start","end"]).reset_index(drop=True)
	return ref_sorted

def check_remove_start_end_same(ref):
	"""
	Remove instance from audio/video reference if the start is the same as end 
	"""
	label_lists = []
	for i in range(0,ref.shape[0]):
		if ref.iloc[i]["start"] == ref.iloc[i]["end"]:
			if ref.iloc[i]["type"] == "text":
				pass
			elif ref.iloc[i]["type"] in ["audio","video"]:
				label_lists.append(i)

	ref.drop(labels=label_lists, inplace = True)

	return ref

def read_dedupe_file(path):
	"""
	Read file and remove duplicate records 
	"""
	df = pd.read_csv(path, dtype={'norm': object}, sep = "\t")
	df.drop_duplicates(inplace = True)

	return df

def get_raw_file_id_dict(file_ids, data_frame, class_type, text_gap, time_gap):
	"""
		Generate a dictionary based on file_id
		
		Parameters
		----------
		file_ids : array of unique file_ids
		data_frame: raw data frame
		class_type: norm or emotion
 
		Returns
		-------
		result_dict: dictionary of file_id
	"""
	data_frame.set_index("file_id")
	result_dict = {}
	for file_id in file_ids:
		sorted_df = extract_df(data_frame, file_id) 
		class_count_vote_dict = get_highest_vote_based_on_time(sorted_df, class_type)

		# Check file type to determine the gap of merging
		if list(sorted_df["type"])[0] == "text" and text_gap is not None:
			vote_array_per_file = merge_vote_time_periods(class_count_vote_dict, text_gap)
		if list(sorted_df["type"])[0] == "text" and text_gap is None:
			vote_array_per_file = merge_vote_time_periods(class_count_vote_dict)		
		if list(sorted_df["type"])[0] != "text" and time_gap is not None:
			vote_array_per_file = merge_vote_time_periods(class_count_vote_dict, time_gap)
		if list(sorted_df["type"])[0] != "text" and time_gap is None:
			vote_array_per_file = merge_vote_time_periods(class_count_vote_dict)		

		result_dict[file_id] = vote_array_per_file
	return result_dict  
		
def get_highest_class_vote(class_dict):
	"""
		Filter out highest voted class/classes with given class dictionary
	 
		Parameters
		----------
		class_dict : dictionary
 
		Returns
		-------
		result_array: list
	"""
	result_array=[]
	high_bar = 2
	for key in class_dict:
		if class_dict[key] >= high_bar:
			result_array.append(key)
	if len(result_array) == 0:
		result_array.append('none')
	return result_array    

def get_highest_vote_based_on_time(data_frame, class_type):
	"""
		Reference Norm/Emotion Instances (after applying Judgment Collapsing by Majority Voting)
		This function should combine class vote in column of "class" in passed in data frame.
		On one particular time frame, this function would count all voted classes and select class/classes get hightest voted.
		e.g.
		From
		Seg1 11s–15s sad,   happy+sad, happy+sad  => 3-sad,   2-happy
		Seg2 16s–18s angry, angry+joy, angry+joy  => 3-angry, 2-joy
		Seg3 19s-23s none,  angry,     joy        => 1-angry, 1-joy
		Seg4 24s-33s joy,   joy,       joy        => 3-joy 
		To result
		Seg1 11s–15s sad+happy 
		Seg2 16s–18s angry+joy 
		Seg3 19s-23s none
		Seg4 24s-33s joy
		
		Parameters
		----------
		data_frame: raw data frame
		class_type: norm or emotion
 
		Returns
		-------
		emo_dict: dictionary which key is norm/emotion and value is start and end time
	"""  
	time_dict = {}
	emo_dict = {}
	pre_key = ''
	voter_count = 0
	counted_items = 0
	for  index,row in data_frame.iterrows():
		# If the start time exists in dictionary, keep voting
		time_key = str(row['start']) + ' - ' + str(row['end']) 
		counted_items = counted_items + 1
		if time_key in time_dict:
			# Keep counting classes
			value = time_dict[time_key]
			if row['user_id'] != value['user_id']:
				voter_count = voter_count + 1
			cur_classes = row['Class'].replace(" ", "").split(",")
			for e in cur_classes:
				if e in value['Class']:
					value['Class'][e] += 1
				else:
					value['Class'][e] = 1
			time_dict[time_key]=value
		else:
			# Finish previous counted vote, only leave the majority counted class for all time periods except the last one
			if pre_key in time_dict:
				pre_value = time_dict[pre_key]
				if voter_count > 1:
					# More than one voter, need to get highest voted class
					highest_vote_class = get_highest_class_vote(pre_value['Class'])
				elif voter_count == 1:
					if class_type == "emotion":
						# Only one voter, translate into noann
						highest_vote_class = [silence_string]
					if class_type == "norm":
						# Only one voter, count his/her votes as result
						highest_vote_class = list(pre_value['Class'].keys())
				time_dict[pre_key]['Class'] = highest_vote_class
				for emo in highest_vote_class:
					pre = pre_key.split(' - ')
					if emo not in emo_dict:
						emo_dict[emo] = []
					emo_dict[emo].append({'start' : pre[0], 'end' :pre[1] })
			# update pre_key
			pre_key = time_key
			voter_count = 1
			# Compose new key value pair
			cur_classes = row['Class'].replace(" ", "").split(",")
			value = {'file_id':row['file_id'], 'segment_id': row['segment_id'], 'start':row['start'], 'end':row['end'], 'user_id':row['user_id'] }
			value['Class'] = {}
			for e in cur_classes:
				value['Class'][e] = 1
			time_dict[time_key] = value
		
		# By the end, add last collected vote to count
		if counted_items == len(data_frame.index):
			cur_value = time_dict[time_key]
			if voter_count > 1:
				# More than one voter, need to get highest voted class
				highest_vote_class = get_highest_class_vote(cur_value['Class'])
			elif voter_count == 1:
				if class_type == "emotion":
					# Only one voter, translate into noann
					highest_vote_class = [silence_string]
				if class_type == "norm":
					# Only one voter, count his/her votes as result
					highest_vote_class = list(cur_value['Class'].keys())
			time_dict[time_key]['Class'] = highest_vote_class  
			for emo in highest_vote_class:
				if emo not in emo_dict:
					emo_dict[emo] = []
				emo_dict[emo].append({'start' : row['start'], 'end' : row['end'] })

	return emo_dict  
	
def merge_vote_time_periods(vote_dict, allowed_gap = None):
	"""
	Merge (time) periods if gap is allowed. allowed_gap is set to 0 by default
	
	Parameters
	----------
	vote_dict : dictionary
	allowed_gap : number

	Returns
	-------
	result_array: list
	"""
	result_array = []
	for key in vote_dict.keys():
		time_array = vote_dict[key]
		# Merge time array
		merged_time_array = []
		i = 0
		while i < len(time_array):
			first_time_period = time_array[i]
			current_time_period = time_array[i]
			if allowed_gap is not None:
				while i + 1 < len(time_array) and float(current_time_period['end']) + allowed_gap > float(time_array[i + 1]['start']):
					i = i + 1
					current_time_period = time_array[i]
			merged_time_array.append({'start': first_time_period['start'], 'end': current_time_period['end']})
			i = i + 1
		for item in merged_time_array:
			result_array.append({'range': item, 'Class': key});
	
	return sorted(result_array, key=lambda emo_range: float(emo_range['range']['start']))

def get_average_dict(file_ids, data_frame):
	"""
		This function should generate a dictionary based on file_id
		
		Parameters
		----------
		file_ids : array of unique file_ids
		data_frame: raw data frame
 
		Returns
		-------
		dictionary of file_id
	"""
	data_frame.set_index("file_id")
	result_dict = {}
	for file_id in file_ids: 
		sorted_df = extract_df(data_frame, file_id)
		average_score_vote_dict = get_average_score_based_on_time(sorted_df)
		result_dict[file_id] = average_score_vote_dict
	return result_dict  

def get_average_score_based_on_time(data_frame):
		"""
		Reference Valence/Arousal Instances (after applying Judgment Averaging)
		e.g.
		From
		Seg1  0s–10s 156, 178, 165
		Seg2 10s–15s 259, 281, 301
		Seg3 15s–18s 978, 899, 950

		To result
		0s-10s 166.3
		10s-15s 280.3
		15s-18s 942.3

		
		Parameters
		----------
		data_frame: raw data frame
 
		Returns
		-------
		time_dict: dictionary which key is period and value includes average valence/arousal value
		"""
		time_dict = {}
		pre_key = ''
		voter_count = 0
		counted_items = 0
		for  index,row in data_frame.iterrows():
			time_key = str(row['start']) + ' - ' + str(row['end']) 
			counted_items = counted_items + 1
			if time_key in time_dict:
				value = time_dict[time_key]
				if row['user_id'] != value['user_id']:
					voter_count = voter_count + 1
				if row['Class'] != silence_string and value['Class'] != silence_string:
					cur_valence = float(row['Class'])
					value['Class'] = value['Class'] + cur_valence
					time_dict[time_key]=value
				else:
					time_dict[time_key]['Class'] = silence_string
			else:
				if pre_key in time_dict:
					pre_value = time_dict[pre_key]
					if voter_count > 1:
						if pre_value['Class'] != silence_string:
							averaged_valence = float(pre_value['Class']) / voter_count
						else:
							averaged_valence = silence_string
					elif voter_count == 1:
						averaged_valence = silence_string
					time_dict[pre_key]['Class'] = averaged_valence
				# update pre_key
				pre_key = time_key
				voter_count = 1
				# Compose new key value pair
			 
				value = {'file_id':row['file_id'], 'segment_id': row['segment_id'], 'start':row['start'], 'end':row['end'], 'user_id':row['user_id'] }
				if row['Class'] != silence_string:
					value['Class'] = float(row['Class'])
				else:
					value['Class'] = silence_string 
				time_dict[time_key] = value
			
			# By the end, add last collected vote to count
			if counted_items == len(data_frame.index):
				cur_value = time_dict[time_key]
				if voter_count > 1:
					if cur_value['Class'] != silence_string:
						averaged_valence = float(cur_value['Class']) / voter_count
						time_dict[pre_key]['Class'] = averaged_valence
					else:
						time_dict[time_key]['Class'] = silence_string
				elif voter_count == 1:
					time_dict[time_key]['Class'] = silence_string
		return time_dict  

def convert_norm_emotion_dict_df(result_dict, class_type):
	"""
	Convert dictionary of norm/emotion into a dataframe
	"""

	file_ids = []
	starts = []
	ends = []
	Class = []

	for file_id in result_dict:
		for segment in result_dict[file_id]:
			file_ids.append(file_id)
			starts.append(float(segment['range']['start']))
			ends.append(float(segment['range']['end']))
			Class.append(segment['Class'])

	result_df = pd.DataFrame({"file_id":file_ids,"Class":Class,"start":starts,"end":ends})
	result_df["Class_type"] = class_type
	return result_df

def convert_valence_arousal_dict_df(result_dict, class_type):
		"""
		Convert dictionary of valence/arousal into a dataframe
		"""

		file_ids = []
		starts = []
		ends = []
		Class = []

		for file_id in result_dict:
			for duration in result_dict[file_id]:
				file_ids.append(result_dict[file_id][duration]["file_id"])
				starts.append(result_dict[file_id][duration]["start"])
				ends.append(result_dict[file_id][duration]["end"])
				Class.append(result_dict[file_id][duration]["Class"])

		result_df = pd.DataFrame({"file_id":file_ids,"Class":Class,"start":starts,"end":ends})
		result_df.drop_duplicates(inplace = True)

		result_df["Class_type"] = class_type
		return result_df

def preprocess_norm_emotion_reference_df(reference_df, class_type, text_gap, time_gap):
	"""
	The wrapper of preprocess for norm/emotion dataframe 
	"""

	new_reference_df = change_class_type(reference_df, class_type)
	# Split input_file into parts based on file_id column
	file_ids = get_unique_items_in_array(new_reference_df['file_id'])
	# Generate file_id map for vote processing
	result = get_raw_file_id_dict(file_ids, new_reference_df, class_type, text_gap, time_gap)
	# Convert the result dictionary into dataframe
	result_df = convert_norm_emotion_dict_df(result, class_type)

	return result_df

def preprocess_valence_arousal_reference_df(reference_df, class_type):
	"""
	The wrappers of preprocess for valence/arousal dataframe 
	"""

	new_reference_df = change_class_type(reference_df, class_type)
	# Split input_file into parts based on file_id column
	file_ids = get_unique_items_in_array(new_reference_df['file_id'])
	# Generate file_id map for vote processing
	result = get_average_dict(file_ids, new_reference_df)
	# Convert the result dictionary into dataframe
	result_df = convert_valence_arousal_dict_df(result, class_type)

	return result_df

def preprocess_reference_dir(ref_dir, scoring_index, task, text_gap, time_gap):
	"""
	For each task, read and merge corresponding data file, segment file and index file
	and then preprocess the merged data frame
	"""
	file_info = pd.read_csv(os.path.join(ref_dir,"docs","file_info.tab"), sep = "\t")
	index_df = file_info.merge(scoring_index, left_on = "file_uid", right_on = "file_id")
	index_df = index_df[["file_id", "type", "length"]]
	index_df.drop_duplicates(inplace = True)

	if task == "norms" or task == "emotions":
		data_file = os.path.join(ref_dir,"data","{}.tab".format(task))
		data_df = read_dedupe_file(data_file)
		data_df = data_df[~data_df.isin(['EMPTY_TBD']).any(axis=1)]  
		segment_file = os.path.join(ref_dir,"docs","segments.tab")
		segment_df = read_dedupe_file(segment_file)
		reference_df = data_df.merge(segment_df.merge(index_df))
		reference_prune = check_remove_start_end_same(reference_df)
		column_name = task.replace("s","")
		ref = preprocess_norm_emotion_reference_df(reference_prune, column_name, text_gap, time_gap)
		ref.drop_duplicates(inplace = True)
		ref_inter = ref.merge(index_df)
		if len(ref_inter) > 0:
			ref_final = fill_start_end(ref_inter)
			ref_final = ref_final[ref_final.Class != "none"]
		else:
			ref_final = ref_inter

	if task == "valence_continuous" or task == "arousal_continuous":
		data_file = os.path.join(ref_dir,"data","valence_arousal.tab")
		data_df = read_dedupe_file(data_file)
		data_df = data_df[~data_df.isin(['EMPTY_TBD']).any(axis=1)]  
		segment_file = os.path.join(ref_dir,"docs","segments.tab")
		segment_df = read_dedupe_file(segment_file)
		reference_df = data_df.merge(segment_df.merge(index_df))
		reference_prune = check_remove_start_end_same(reference_df)
		column_name = task
		ref = preprocess_valence_arousal_reference_df(reference_prune, column_name)
		ref = ref.merge(index_df)
		ref_inter = extend_gap_segment(ref)
		if len(ref_inter) > 0:
			ref_final = fill_start_end(ref_inter)
		else:
			ref_final = ref_inter		

	if task == "changepoint":
		data_file = os.path.join(ref_dir,"data","{}.tab".format(task))
		data_df = read_dedupe_file(data_file)
		data_df = data_df[~data_df.isin(['EMPTY_TBD']).any(axis=1)]  
		ref = data_df.merge(index_df)
		ref = ref[ref.timestamp != "none"]
		ref_final = change_class_type(ref, convert_task_column(task))

	return ref_final




