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

def make_row(arr):
        if (arr[7] is None): 
                return(pd.Series(arr[0:7], index=['file_id', 'Class', 'start', 'end', 'Class_type', 'type', 'length']))
        else:
                return(pd.Series(arr[0:8], index=['file_id', 'Class', 'start', 'end', 'Class_type', 'type', 'length', 'status']))

def fill_start_end_ref(ref):
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
	annot_segments: dictionary of the start and end of the annotation

	"""
	
	class_type = list(ref["Class_type"])[0]
	file_ids = get_unique_items_in_array(ref['file_id'])
	### Check if there is a status field.  If so, then the adjust the array to have an empyt status
	has_status = ('status' in ref.columns)
	for i in file_ids:
		sub_ref = extract_df(ref, i)
		type = list(sub_ref["type"])[0]
		length = list(sub_ref["length"])[0]
		start = min(list(sub_ref["start"]))
		end = max(list(sub_ref["end"]))
		status = "noann" if (has_status) else None
                
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
	return(ref_sorted)

def file_based_merge_ref(ref, annot_segments, file_merge_proportion):
	"""
	Merge to the full extent of the annotated region.

	Parameters
	----------
	ref
        text_gap
        time_gap

	Returns
	-------
	ref data frame after merging to file-based scoring

	"""
	
	class_type = list(ref["Class_type"])[0]
	file_ids = get_unique_items_in_array(ref['file_id'])
	#print(f"gaps: time={time_gap}  text={text_gap}")

	### Check if there is a status field.  If so, then the adjust the array to have an empyt status
	list_of_lists = []
	for file_id in file_ids:
		sub = ref[(ref['file_id'] == file_id) & (ref['Class'] != 'noann:seg')]
		type = list(sub["type"])[0]
		length = list(sub["length"])[0]
		sub_annot_segments = annot_segments.loc[annot_segments["file_id"] == file_id]
		step = (max(sub_annot_segments["end"]) - min(sub_annot_segments["start"]))*file_merge_proportion
		start = list(np.around(np.array(list(sub_annot_segments["start"])),3))
		end = list(np.around(np.array(list(sub_annot_segments["end"])),3))
		time_pool = start + end
		if (max(time_pool) - min(time_pool)) % step == 0:
			step_list = np.linspace(min(time_pool), max(time_pool), int((max(time_pool) - min(time_pool))/step + 1))
		else:
			divisor = 1/file_merge_proportion
			remainder = (max(time_pool) - min(time_pool)) - int(step)*divisor
			step_list = np.append(np.linspace(min(time_pool), max(time_pool)-remainder-int(step), int(divisor)), max(time_pool))

		startpoint = []
		endpoint = []
		temp_step = {}
		final_step = {}
		temp_label = {}

		for i in range(len(step_list)-1):
			startpoint.append(round(step_list[i],3))
			endpoint.append(round(step_list[i+1],3))
			temp_step[i] = []
			temp_label[i] = []

		count = 0
		for i,j in zip(startpoint,endpoint):
			for k in range(sub_annot_segments.shape[0]):
				if i <= start[k] and i <= end[k] and j >= end[k]:
					temp_step[count].extend([float(start[k]), float(end[k])])
					temp_label[count].extend(list(sub.loc[(sub["start"] == start[k]) & (sub["end"] == end[k]), "Class"].values))
				if i <= start[k] and i < end[k] and j < end[k] and start[k] < j:
					left_overlap = abs(j - start[k])
					right_overlap = abs(end[k] - j)
					if left_overlap >= right_overlap:
						temp_step[count].extend([float(start[k]), float(end[k])])
						temp_label[count].extend(list(sub.loc[(sub["start"] == start[k]) & (sub["end"] == end[k]), "Class"].values))
					else:
						temp_step[count+1].extend([float(start[k]), float(end[k])])
						temp_label[count+1].extend(list(sub.loc[(sub["start"] == start[k]) & (sub["end"] == end[k]), "Class"].values))
			count = count + 1

		for i in range(len(temp_step)):
			if len(temp_step[i]) > 0:
				final_step[i] = [min(temp_step[i]), max(temp_step[i])]
		
		final_label = {}
		for i in range(len(temp_label)):
			if len(temp_step[i]) > 0:
				noann_prec = temp_label[i].count("noann")/len(temp_label[i])
				if noann_prec >= 0.5:
					final_label[i] = ["noann"]
				else:
					final_label[i] = list(set([item for item in temp_label[i] if item != "noann"]))
		for i in list(final_step.keys()):
			if len(final_label[i]) > 0:
				for j in final_label[i]:
					list_of_lists.append([file_id,j,final_step[i][0],final_step[i][1],class_type,type,length])

	new_ref = pd.DataFrame(list_of_lists, columns=['file_id', 'Class', 'start', 'end', 'Class_type', 'type', 'length'])
	new_seg = new_ref[["file_id","start","end"]].drop_duplicates()
	return new_ref, new_seg			
			
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

def get_raw_file_id_dict(file_ids, data_frame, class_type, text_gap, time_gap, merge_label, minimum_vote_agreement):
	"""
		Generate a dictionary based on file_id
		
		Parameters
		----------
		file_ids : array of unique file_ids
		data_frame: raw data frame
		class_type: norm or emotion
		norm_status: norm's status.  Omitted for emotions
                minimum_vote_agreement : integer The mimimum agreement between annotators
 
		Returns
		-------
		result_dict: dictionary of file_id
	"""
	data_frame.set_index("file_id")
	result_dict = {}
	for file_id in file_ids:
		sorted_df = extract_df(data_frame, file_id)
		class_count_vote_dict = get_highest_vote_based_on_time(sorted_df, class_type, minimum_vote_agreement)                
		# Check file type to determine the gap of merging
		if list(sorted_df["type"])[0] == "text" and text_gap is not None:
			vote_array_per_file = merge_vote_time_periods(class_count_vote_dict, class_type, text_gap, merge_label=merge_label)
		if list(sorted_df["type"])[0] == "text" and text_gap is None:
			vote_array_per_file = merge_vote_time_periods(class_count_vote_dict, class_type, merge_label=merge_label)		
		if list(sorted_df["type"])[0] != "text" and time_gap is not None:
			vote_array_per_file = merge_vote_time_periods(class_count_vote_dict, class_type, time_gap, merge_label=merge_label)
		if list(sorted_df["type"])[0] != "text" and time_gap is None:
			vote_array_per_file = merge_vote_time_periods(class_count_vote_dict, class_type, merge_label=merge_label)		
                                        
		result_dict[file_id] = vote_array_per_file              

	return result_dict  
		
def get_highest_class_vote(class_dict, minimum_vote_agreement):
	"""
		Filter out highest voted class/classes with given class dictionary
	 
		Parameters
		----------
		class_dict : dictionary
                minimum_vote_agreement : integer The mimimum agreement between annotators
 
		Returns
		-------
		result_array: list
	"""
	result_array=[]
	for key in class_dict:
		if class_dict[key] >= minimum_vote_agreement:
			result_array.append(key)
	if len(result_array) == 0:
		result_array.append('none')
	return result_array

def pick_highest_vote(voter_count, minimum_vote_agreement, value, class_type):

	if class_type == "emotion" and minimum_vote_agreement > 1:
		if voter_count >= minimum_vote_agreement:
			highest_vote_class = get_highest_class_vote(value['Class'], minimum_vote_agreement)
		elif voter_count < minimum_vote_agreement:
				highest_vote_class = [silence_string]

	else:
		highest_vote_class = list(value['Class'].keys())

	return highest_vote_class

def get_highest_vote_based_on_time(data_frame, class_type, minimum_vote_agreement):
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
	for index,row in data_frame.iterrows():
		# If the start time exists in dictionary, keep voting
		time_key = str(row['start']) + ' - ' + str(row['end'])                
		counted_items = counted_items + 1
		if time_key in time_dict and class_type != "norm":  #### OK, so!  norms are never multi-annotated.  if they are THIS WILL FAIL!!!!!!
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
			if (class_type == "norm"):
				if row['status'] not in value['status']:
								value['status'][row['status']] = 0
				value['status'][row['status']] += 1
			time_dict[time_key]=value
		else:
			# Finish previous counted vote, only leave the majority counted class for all time periods except the last one
			if pre_key in time_dict:
				pre_value = time_dict[pre_key]
				highest_vote_class = pick_highest_vote(voter_count, minimum_vote_agreement, pre_value, class_type)
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
			# update pre_key
			pre_key = time_key
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
			highest_vote_class = pick_highest_vote(voter_count, minimum_vote_agreement, cur_value, class_type)
			time_dict[time_key]['Class'] = highest_vote_class  
			for emo in highest_vote_class:
				if emo not in emo_dict:
					emo_dict[emo] = []
				em = {'start' : row['start'], 'end' : row['end'] }
				if (class_type == "norm"):
					assert len(cur_value['status'].keys()) == 1, "Error: Multiple norm judgements with differing status"
					em['status'] = list(cur_value["status"])[0]
				emo_dict[emo].append(em)

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
		time_array = vote_dict[key]
		# Merge time array
		merged_time_array = []
		i = 0
		while i < len(time_array):                        
			first_time_period = time_array[i]
			current_time_period = time_array[i]
			if (class_type == "norm"):
				status_dict = {}
				status_dict[current_time_period['status']] = 1  ### add the first one
			if allowed_gap is not None and key != "noann":
				while ((i + 1 < len(time_array)) and 
                                       (float(current_time_period['end']) + allowed_gap > float(time_array[i + 1]['start'])) and
                                       ((class_type == "emotion") or
                                        ((class_type == "norm") and
                                         ((merge_label is None or merge_label == 'class') or
                                          ((merge_label == 'class-status') and (current_time_period['status'] == time_array[i + 1]['status'])))))):
					if (class_type == "norm"):
						status_dict[time_array[i+1]['status']] = 1
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

def preprocess_norm_emotion_reference_df(reference_df, class_type, text_gap, time_gap, merge_label, minimum_vote_agreement):
	"""
	The wrapper of preprocess for norm/emotion dataframe 

        minimum_vote_agreement : integer The mimimum agreement between annotators

        """

	new_reference_df = change_class_type(reference_df, class_type)
	# Split input_file into parts based on file_id column
	file_ids = get_unique_items_in_array(new_reference_df['file_id'])
	# Generate file_id map for vote processing
	result = get_raw_file_id_dict(file_ids, new_reference_df, class_type, text_gap, time_gap, merge_label, minimum_vote_agreement)        
	#print("Before")
	#print(result)
        
	# Convert the result dictionary into dataframe
	result_df = convert_norm_emotion_dict_df(result, class_type)
	#print("After")
	#print(result_df)

	# ## Dropping rows with zero duration
	# result_df = result_df.drop(result_df[result_df.start == result_df.end].index, axis=0)

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

def convert_reference_zscore(df, task):

	df_num = df[df[task] != "noann"].copy()
	df_num[task] = df_num[task].astype(int)

	Mean = df_num.groupby(['user_id'])[task].transform('mean')    
	Std = df_num.groupby(['user_id'])[task].transform('std').fillna(0)

	df_num["zscore"] = ((df_num[task] - Mean)/Std).fillna(0)
	df_noann = df[df[task] == "noann"]
	df_zscore = df_num.drop([task], axis=1)
	df_zscore_final = df_zscore.rename(columns={"zscore":task})
	df_zscore_final = pd.concat([df_noann, df_zscore_final]).sort_values(by = ["user_id","segment_id"])

	return df_zscore_final

def fix_ref_status_conflict(df):

	df_nonone = df.loc[df["norm"] != "none"]
	status_counts = df_nonone.groupby(["user_id","file_id","segment_id","norm"]).size().reset_index(name='counts')
	status_counts_filtered = status_counts.loc[status_counts["counts"] > 1]

	for index, row in status_counts_filtered.iterrows():
		df.loc[(df["user_id"] == row["user_id"]) & (df["file_id"] == row["file_id"]) & (df["segment_id"] == row["segment_id"]), "status"] = "noann"
		df.loc[(df["user_id"] == row["user_id"]) & (df["file_id"] == row["file_id"]) & (df["segment_id"] == row["segment_id"]), "norm"] = "noann"

	return df

### If text_gap or time_gap is 9999999999, then it is a file-based mergeing for ND and ED!!!
def preprocess_reference_dir(ref_dir, scoring_index, task, text_gap = None, time_gap = None, merge_label = None, dump_inputs = False, output_dir = None, fix_ref_status_conflict_label = None, minimum_vote_agreement = None, file_merge_proportion = None):
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
		if (fix_ref_status_conflict_label) and (task == 'norms'):
			data_df = fix_ref_status_conflict(data_df)
		segment_file = os.path.join(ref_dir,"docs","segments.tab")
		segment_df = read_dedupe_file(segment_file)
		reference_df = data_df.merge(segment_df.merge(index_df))

		if (dump_inputs):
			reference_df.to_csv(os.path.join(output_dir, "inputs.ref.read.tab"), sep = "\t", index = None)
		reference_prune = check_remove_start_end_same(reference_df)
		column_name = task.replace("s","")

		if (task == 'norms'):
			reference_prune['norm_status'] = [ n + "::" + s for n,s in zip(reference_prune['norm'], reference_prune['status']) ]
			#Begin to keep the status here.  Stopping for now
		ref = preprocess_norm_emotion_reference_df(reference_prune, column_name, text_gap, time_gap, merge_label, minimum_vote_agreement)
                
		ref.drop_duplicates(inplace = True)
		ref_inter = ref.merge(index_df)

		if len(ref_inter) > 0:
			if file_merge_proportion:
				ref_final, ref_seg = file_based_merge_ref(ref_inter, segment_df, file_merge_proportion)
				return ref_final, ref_seg
			else:
				ref_final = fill_start_end_ref(ref_inter)   ### This adds the beginning and ending noann segments
				# ref_final = ref_inter
		else:
			ref_final = ref_inter

	if task == "valence_continuous" or task == "arousal_continuous":
		data_file = os.path.join(ref_dir,"data","valence_arousal.tab")
		data_df = read_dedupe_file(data_file)
		data_df = data_df[~data_df.isin(['EMPTY_TBD']).any(axis=1)] 
		new_data_df = convert_reference_zscore(data_df, task)
		segment_file = os.path.join(ref_dir,"docs","segments.tab")
		segment_df = read_dedupe_file(segment_file)
		reference_df = new_data_df.merge(segment_df.merge(index_df))
		reference_prune = check_remove_start_end_same(reference_df)
		column_name = task
		ref = preprocess_valence_arousal_reference_df(reference_prune, column_name)
		ref = ref.merge(index_df)
		ref_inter = extend_gap_segment(ref, "ref")
		if len(ref_inter) > 0:
			ref_final = fill_start_end_ref(ref_inter)
		else:
			ref_final = ref_inter		

	### Change
	if task == "changepoint":
		data_file = os.path.join(ref_dir,"data","{}.tab".format(task))
		data_df = read_dedupe_file(data_file)
		data_df = data_df[~data_df.isin(['EMPTY_TBD']).any(axis=1)]  

		if "direction_impact" in list(data_df.columns):
			data_df = data_df.drop("direction_impact", axis=1)
		if "strength_impact" in list(data_df.columns):
			data_df.rename(columns={'strength_impact':'impact'}, inplace=True)
		if "impact_scalar" in list(data_df.columns):
			data_df.rename(columns={'impact_scalar':'impact'}, inplace=True)		

		ref = data_df.merge(index_df)
		ref = ref[ref.timestamp != "none"]
		ref['start'] = ref['timestamp']  ### Add the timestamp to handle noscore segments with duration
		ref['end'] = ref['timestamp']  ### Add the timestamp to handle noscore segments with duration

		ref_final = change_class_type(ref, convert_task_column(task))  ### Changes the timestamp to Class

		if ref_final.shape[0] > 0:
			## Load the segment file to add Noscore segments
			segment_file = os.path.join(ref_dir,"docs","segments.tab")
			segment_df = read_dedupe_file(segment_file)
			segment_df = segment_df.merge(index_df)

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
									ref_final.loc[len(ref_final)] = [0, file_id, 0,   'NO_SCORE_REGION', "StartNoScore", type_col, length, 0,   start]
					if (end < length):
									ref_final.loc[len(ref_final)] = [0, file_id, end, 'NO_SCORE_REGION', "EndNoScore",   type_col, length, end, length]
									
	return ref_final




