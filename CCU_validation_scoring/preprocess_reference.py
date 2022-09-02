import os
from time import time
import pandas as pd
from .utils import *

silence_string = "nospeech"

def process_subset_norm_emotion(list_file, ref):
	""" 
	Method to check filter args and apply them to ref so they can be used by
	multiple commands.    
	"""
	norm_emotion_list = load_list(list_file)
	# Exclude all classes but include relevant norm in reference
	pruned_ref = ref.loc[(ref.Class.isin(norm_emotion_list)) | (ref.Class == silence_string)]
	
	return pruned_ref

def read_dedupe_reference_file(path):
	"""
	Read file and remove duplicate records 
	"""
	df = pd.read_csv(path, sep = "\t")
	df.drop_duplicates(inplace = True)

	return df

def delete_gap_segment(segment_df, gap = 0.001):
	"""
	Remove the gap between end of previous segment and start of next segment in LDC segments.tab. The gap is set to 0.001 by default

	e.g.
	From
	Seg  Start  End
	Seg1 45.2   60.2
	Seg2 60.201 75.201
	To result
	Seg  Start  End
	Seg1 45.2   60.2
	Seg2 60.2   75.2

	Parameters
	----------
	segment_df, gap

	Returns
	-------
	new_segment_df: data frame after removing the gap

	"""

	new_start = list(segment_df["start"])
	new_end = list(segment_df["end"])

	for i in range(1,segment_df.shape[0]):

		diff = round(segment_df.iloc[i]["start"] - segment_df.iloc[i-1]["end"],3)
		if diff == gap and segment_df.iloc[i]["file_id"] == segment_df.iloc[i-1]["file_id"]:
			new_start[i] = round(new_end[i-1],3)
			step = segment_df.iloc[i]["end"] - segment_df.iloc[i]["start"]
			new_end[i] = round(new_start[i]+step,3)

	segment_df["new_start"] = new_start
	segment_df["new_end"] = new_end

	segment_df = segment_df[["file_id","segment_id","new_start","new_end"]]
	new_segment_df = segment_df.rename(columns={"new_start": "start", "new_end": "end"})

	return new_segment_df


def get_unique_items_in_array(file_id_array):
	"""
		Extract unique items from an array and return in array format
	 
		Parameters
		----------
		file_id_array : array
 
		Returns
		-------
		list
	"""
	return list(set(file_id_array))

def get_raw_file_id_dict(file_ids, data_frame):
	"""
		Generate a dictionary based on file_id
		
		Parameters
		----------
		file_ids : array of unique file_ids
		data_frame: raw data frame
 
		Returns
		-------
		result_dict: dictionary of file_id
	"""
	data_frame.set_index("file_id")
	result_dict = {}
	for file_id in file_ids:
		sorted_df = extract_df(data_frame, file_id) 
		class_count_vote_dict = get_highest_vote_based_on_time(sorted_df)
		# Check file type to determine the gap of merging
		if list(sorted_df["type"])[0] == "text":
			gap = 10
		else:
			gap = 1

		vote_array_per_file = merge_vote_time_periods(class_count_vote_dict, gap)
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
	return result_array    
			
		
def get_highest_vote_based_on_time(data_frame):
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
			cur_classes = row['Class'].split(', ')
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
			cur_classes = row['Class'].split(', ')
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
				# Only one voter, count his/her votes as result
				highest_vote_class = list(cur_value['Class'].keys())
			time_dict[time_key]['Class'] = highest_vote_class  
			for emo in highest_vote_class:
					if emo not in emo_dict:
						emo_dict[emo] = []
					emo_dict[emo].append({'start' : row['start'], 'end' : row['end'] })

	return emo_dict  
	
def merge_vote_time_periods(vote_dict, allowed_gap = 0):
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
				if row['Class'] != silence_string:
					cur_valence = float(row['Class'])
					value['Class'] = value['Class'] + cur_valence
					time_dict[time_key]=value
				else:
					time_dict[time_key]['Class'] = silence_string
			else:
				if pre_key in time_dict:
					pre_value = time_dict[pre_key]
					if pre_value['Class'] != silence_string:
						averaged_valence = float(pre_value['Class']) / voter_count
					else:
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
				if cur_value['Class'] != silence_string:
					averaged_valence = float(cur_value['Class']) / voter_count
					time_dict[pre_key]['Class'] = averaged_valence
				else:
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

def preprocess_norm_emotion_reference_df(reference_df, class_type):
	"""
	The wrapper of preprocess for norm/emotion dataframe 
	"""

	new_reference_df = change_class_type(reference_df, class_type)
	# Split input_file into parts based on file_id column
	file_ids = get_unique_items_in_array(new_reference_df['file_id'])
	# Generate file_id map for vote processing
	result = get_raw_file_id_dict(file_ids, new_reference_df)
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

def preprocess_reference_dir(ref_dir, task):
	"""
	For each task, read and merge corresponding data file, segment file and index file(read this from reference directory for now) 
	and then preprocess the merged data frame
	"""

	index_file = os.path.join(ref_dir,"docs","system_input.index.tab")
	index_df = read_dedupe_reference_file(index_file)

	if task == "norms" or task == "emotions":
		data_file = os.path.join(ref_dir,"data","{}.tab".format(task))
		data_df = read_dedupe_reference_file(data_file)  
		segment_file = os.path.join(ref_dir,"docs","segments.tab")
		segment_df = read_dedupe_reference_file(segment_file)
		segment_prune = delete_gap_segment(segment_df)
		reference_df = data_df.merge(segment_prune.merge(index_df))
		column_name = task.replace("s","")
		ref = preprocess_norm_emotion_reference_df(reference_df, column_name)
		ref = ref[ref.Class != "none"]
		ref = ref.merge(index_df)

	if task == "valence_continuous" or task == "arousal_continuous":
		data_file = os.path.join(ref_dir,"data","valence_arousal.tab")
		data_df = read_dedupe_reference_file(data_file)  
		segment_file = os.path.join(ref_dir,"docs","segments.tab")
		segment_df = read_dedupe_reference_file(segment_file)
		segment_prune = delete_gap_segment(segment_df)
		reference_df = data_df.merge(segment_prune.merge(index_df))
		column_name = task
		ref = preprocess_valence_arousal_reference_df(reference_df, column_name)
		ref = ref.merge(index_df)

	if task == "changepoint":
		data_file = os.path.join(ref_dir,"data","{}.tab".format(task))
		data_df = read_dedupe_reference_file(data_file)  
		ref = data_df.merge(index_df)
		ref = ref[ref.timestamp != "none"]
		ref = change_class_type(ref, convert_task_column(task))

	return ref




