import os
import pprint
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

def make_row(arr):
        if (arr[7] is None): 
                return(pd.Series(arr[0:7], index=['file_id', 'Class', 'start', 'end', 'Class_type', 'type', 'length']))
        else:
                return(pd.Series(arr[0:8], index=['file_id', 'Class', 'start', 'end', 'Class_type', 'type', 'length', 'status']))

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
	### Check if there is a status field.  If so, then the adjust the array to have an empyt status
	#print(ref)
	has_status = ('status' in ref.columns)
	for i in file_ids:
		sub_ref = extract_df(ref, i)
		type = list(sub_ref["type"])[0]
		length = list(sub_ref["length"])[0]
		start = min(list(sub_ref["start"]))
		end = max(list(sub_ref["end"]))
		status = list(sub_ref["status"])[0] if (has_status) else None
                
		if type == "text":
			if start > 0:
				ref.loc[len(ref.index)] = make_row([i, silence_string, 0, start - 1, class_type, type, length, status])
			if end < length:
				ref.loc[len(ref.index)] = make_row([i, silence_string, end + 1, length, class_type, type, length, status])

		if type in ["audio","video"]:
			if start > 0:
				ref.loc[len(ref.index)] = make_row([i, silence_string, 0, start, class_type, type, length, status])
			if end < length:
				ref.loc[len(ref.index)] = make_row([i, silence_string, end, length, class_type, type, length, status])
	
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

def get_raw_file_id_dict(file_ids, data_frame, class_type, text_gap, time_gap, merge_label):
	"""
		Generate a dictionary based on file_id
		
		Parameters
		----------
		file_ids : array of unique file_ids
		data_frame: raw data frame
		class_type: norm or emotion
		norm_status: norm's status.  Omitted for emotions
 
		Returns
		-------
		result_dict: dictionary of file_id
	"""
	data_frame.set_index("file_id")
	result_dict = {}
	for file_id in file_ids:
		sorted_df = extract_df(data_frame, file_id)
		#print(class_type)
		#if (class_type == 'norm'):  ### Status needs to be serviced
		#        sorted_df['Class'] = sorted_df['norm_status']
		#print("Ref data to merge")
		#print(sorted_df)
		class_count_vote_dict = get_highest_vote_based_on_time(sorted_df, class_type)
		#print("\nClass count before time vote")
		#print(pprint.pprint(class_count_vote_dict))
                
		# Check file type to determine the gap of merging
		if list(sorted_df["type"])[0] == "text" and text_gap is not None:
			vote_array_per_file = merge_vote_time_periods(class_count_vote_dict, class_type, text_gap, merge_label=merge_label)
		if list(sorted_df["type"])[0] == "text" and text_gap is None:
			vote_array_per_file = merge_vote_time_periods(class_count_vote_dict, class_type, merge_label=merge_label)		
		if list(sorted_df["type"])[0] != "text" and time_gap is not None:
			vote_array_per_file = merge_vote_time_periods(class_count_vote_dict, class_type, time_gap, merge_label=merge_label)
		if list(sorted_df["type"])[0] != "text" and time_gap is None:
			vote_array_per_file = merge_vote_time_periods(class_count_vote_dict, class_type, merge_label=merge_label)		
                                        
		#print("Vote array after time merge")
		#print(pprint.pprint(vote_array_per_file, width=200))
		#exit(0)
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

def get_highest_vote_based_on_time_orig(data_frame, class_type):
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

	#print(pprint.pprint(emo_dict))
	return emo_dict  

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
	#print("Get_highest")
	#print(data_frame)
	for index,row in data_frame.iterrows():
		# If the start time exists in dictionary, keep voting
		time_key = str(row['start']) + ' - ' + str(row['end'])                
		#print("PRocess {}".format(time_key))
		counted_items = counted_items + 1
		if time_key in time_dict and class_type != "norm":  #### OK, so!  norms are never multi-annotated.  if they are THIS WILL FAIL!!!!!!
			# Keep counting classes
			#print(f"  Appending additional judgement")
			value = time_dict[time_key]
			if row['user_id'] != value['user_id']:
				voter_count = voter_count + 1
			cur_classes = row['Class'].replace(" ", "").split(",")
			for e in cur_classes:
				if e in value['Class']:
					value['Class'][e] += 1
				else:
					value['Class'][e] = 1
			if (class_type == "norm"):
			        if row['status'] not in value['status']:
			                value['status'][row['status']] = 0
			        value['status'][row['status']] += 1
			time_dict[time_key]=value
			#print(f"    value{value}")
		else:
			# Finish previous counted vote, only leave the majority counted class for all time periods except the last one
			if pre_key in time_dict:
				pre_value = time_dict[pre_key]
				#print("  Finalize dup judgement {}".format(pre_value))
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
					em = {'start' : pre[0], 'end' :pre[1] }
					if (class_type == "norm"):
                                                assert len(pre_value['status'].keys()) == 1, "Error: Multiple norm judgements with differing status"
                                                em['status'] = list(pre_value["status"])[0]
					emo_dict[emo].append(em)
					#print(f"    Final emo_dict {emo} {em}")
			# update pre_key
			pre_key = time_key
			#print(f"  Start {pre_key}")
			voter_count = 1
			# Compose new key value pair
			cur_classes = row['Class'].replace(" ", "").split(",")
			value = {'file_id':row['file_id'], 'segment_id': row['segment_id'], 'start':row['start'], 'end':row['end'], 'user_id':row['user_id'] }
			value['Class'] = {}
			for e in cur_classes:
				value['Class'][e] = 1
			if (class_type == "norm"):
			        value['status'] = {}
			        value['status'][row['status']] = 1
			time_dict[time_key] = value
		
		# By the end, add last collected vote to count
		if counted_items == len(data_frame.index):
			cur_value = time_dict[time_key]
			#print(f"Final items to finalize {cur_value}")
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
				em = {'start' : row['start'], 'end' : row['end'] }
				if (class_type == "norm"):
				        assert len(cur_value['status'].keys()) == 1, "Error: Multiple norm judgements with differing status"
				        em['status'] = list(cur_value["status"])[0]
				emo_dict[emo].append(em)

	#print("emo_dict")
	#print(pprint.pprint(emo_dict))
	return emo_dict  
	
def merge_vote_time_periods(vote_dict, class_type, allowed_gap = None, merge_label = None):
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
		#print("merge for key {}".format(key))
		time_array = vote_dict[key]
		# Merge time array
		merged_time_array = []
		i = 0
		while i < len(time_array):                        
			first_time_period = time_array[i]
			#print("ftp={} i={} start={} merge_label={} allowed_gap={}".format(first_time_period, i, vote_dict[key][i]['start'], merge_label, allowed_gap))
			current_time_period = time_array[i]
			if (class_type == "norm"):
                                status_dict = {}
                                status_dict[current_time_period['status']] = 1  ### add the first one
			if allowed_gap is not None:
				#if (i + 1 < len(time_array)):
				#        print(f"  test {current_time_period['end']} and {time_array[i + 1]['start']}")
				while ((i + 1 < len(time_array)) and 
                                       (float(current_time_period['end']) + allowed_gap > float(time_array[i + 1]['start'])) and
                                       ((class_type == "emotion") or
                                        ((class_type == "norm") and
                                         ((merge_label is None or merge_label == 'class') or
                                          ((merge_label == 'class-status') and (current_time_period['status'] == time_array[i + 1]['status'])))))):
					if (class_type == "norm"):
					        #print(f"  norm merge {i}+1 {time_array[i+1]} status={status_dict}")
					        status_dict[time_array[i+1]['status']] = 1
					#elif (class_type == "emotion"):
					        #print(f"  emotion merge {i}+1 {time_array[i+1]}")
					i = i + 1
					current_time_period = time_array[i]
			info = {'Class': key, 'range': {'start': first_time_period['start'], 'end': current_time_period['end']}}
			if (class_type == "norm"):
                                st = list(status_dict.keys())
                                st.sort()
                                info['status'] = ','.join(st)
                        
			merged_time_array.append(info)
			i = i + 1
		for item in merged_time_array:
			result_array.append(item)
	
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
	status = []

	for file_id in result_dict:
		for segment in result_dict[file_id]:
			file_ids.append(file_id)
			starts.append(float(segment['range']['start']))
			ends.append(float(segment['range']['end']))
			Class.append(segment['Class'])
			if (class_type == "norm"):
			        status.append(segment['status'])

	result_df = pd.DataFrame({"file_id":file_ids,"Class":Class,"start":starts,"end":ends})
	if (class_type == "norm"):
                result_df["status"] = status
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

def preprocess_norm_emotion_reference_df(reference_df, class_type, text_gap, time_gap, merge_label):
	"""
	The wrapper of preprocess for norm/emotion dataframe 
	"""

	#print("begin convert {}",class_type)
	#print(reference_df)
	new_reference_df = change_class_type(reference_df, class_type)
	# Split input_file into parts based on file_id column
	file_ids = get_unique_items_in_array(new_reference_df['file_id'])
	# Generate file_id map for vote processing
	result = get_raw_file_id_dict(file_ids, new_reference_df, class_type, text_gap, time_gap, merge_label)
	#print("raw_dict time={} text={}".format(time_gap, text_gap))
	#print(pprint.pprint(result, width=200))
	# Convert the result dictionary into dataframe
	result_df = convert_norm_emotion_dict_df(result, class_type)

	## Dropping rows with zero duration
	result_df = result_df.drop(result_df[result_df.start == result_df.end].index, axis=0)
	#print(result_df)

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

def preprocess_reference_dir(ref_dir, scoring_index, task, text_gap = None, time_gap = None, merge_label = None, dump_inputs = False, output_dir = None):
	"""
	For each task, read and merge corresponding data file, segment file and index file
	and then preprocess the merged data frame
	"""
	file_info = pd.read_csv(os.path.join(ref_dir,"docs","file_info.tab"), sep = "\t")
	index_df = file_info.merge(scoring_index, left_on = "file_uid", right_on = "file_id")
	index_df = index_df[["file_id", "type", "length"]]
	index_df.drop_duplicates(inplace = True)

	pd.set_option('display.max_columns', None)
	pd.set_option('display.width', 1000)
	if task == "norms" or task == "emotions":
		data_file = os.path.join(ref_dir,"data","{}.tab".format(task))
		data_df = read_dedupe_file(data_file)
		data_df = data_df[~data_df.isin(['EMPTY_TBD']).any(axis=1)]  
		segment_file = os.path.join(ref_dir,"docs","segments.tab")
		segment_df = read_dedupe_file(segment_file)
		reference_df = data_df.merge(segment_df.merge(index_df))
		if (dump_inputs):
                        reference_df.to_csv(os.path.join(output_dir, "inputs.ref.read.tab"), sep = "\t", index = None)
		reference_prune = check_remove_start_end_same(reference_df)
		column_name = task.replace("s","")
		if (task == 'norms'):
                        reference_prune['norm_status'] = [ n + "::" + s for n,s in zip(reference_prune['norm'], reference_prune['status']) ]
                        #column_name = 'norm_status'
                        #Begin to keep the status here.  Stopping for now
		ref = preprocess_norm_emotion_reference_df(reference_prune, column_name, text_gap, time_gap, merge_label)
		#print("final")
		#print(ref)
		#exit(0)
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

	### Change
	if task == "changepoint":
		data_file = os.path.join(ref_dir,"data","{}.tab".format(task))
		data_df = read_dedupe_file(data_file)
		data_df = data_df[~data_df.isin(['EMPTY_TBD']).any(axis=1)]  
		ref = data_df.merge(index_df)
		ref = ref[ref.timestamp != "none"]
		ref['start'] = ref['timestamp']  ### Add the timestamp to handle noscore segments with duration
		ref['end'] = ref['timestamp']  ### Add the timestamp to handle noscore segments with duration
		#print("before class change")
		#printz(ref)
		ref_final = change_class_type(ref, convert_task_column(task))  ### Changes the timestamp to Class
		#print("Ref before tweaks")
		#print(ref_final)
		## Load the segment file to add Noscore segments
		segment_file = os.path.join(ref_dir,"docs","segments.tab")
		segment_df = read_dedupe_file(segment_file)
		segment_df = segment_df.merge(index_df)
		#print("Segment file")
		#print(segment_df)
		#Add the noscore regions 
		if (True):
		        for file_id in set(segment_df['file_id']):
                                #print("File {}".format(file_id))
                                ref_sub = segment_df[segment_df['file_id'] == file_id]
                                #print(ref_sub)
                                start = min(list(ref_sub["start"]))
                                end = max(list(ref_sub["end"]))
                                length = ref_sub.loc[ref_sub.index[0], 'length'] 
                                type_col = ref_sub.loc[ref_sub.index[0], 'type'] 
                                #print("file {} {} {} {}".format(file_id, start, end, length))
                                if (start > 0):
                                        ref_final.loc[len(ref_final)] = [0, file_id, 0,   'NO_SCORE_REGION', "StartNoScore", type_col, length, 0, start]
                                if (end < length):
                                        ref_final.loc[len(ref_final)] = [0, file_id, end, 'NO_SCORE_REGION', "EndNoScore", type_col, length, end, length]

	ref_final['ref_uid'] = [ "R"+str(s) for s in range(len(ref_final['file_id'])) ] ### This is a unique REF ID to help find FN 
	#print("DONE - ref_final")
	#print(ref_final)
	#exit(0)
	return ref_final




