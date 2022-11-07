import os
import pandas as pd
import numpy as np
from .utils import *
from .preprocess_reference import *
         
silence_string = "noann"

def change_continuous_text(df):
		"""
		Convert the ref and hyp into discrete time-series (Text-based decision unit)
		e.g.
		From 
		docid	start end Class
		doc1	1	  2	  1
		To
		docid	point Class
		doc1	1	  1
		doc1	2	  1
		"""
		
		doc = []
		point = []
		label_new = []

		df["diff"] = list(abs(df["end"] - df["start"]) + 1)

		for i in range(0,df.shape[0]):
			doc.extend([df.iloc[i]["file_id"]] * int(df.iloc[i]["diff"]))
			point.extend(np.linspace(df.iloc[i]["start"],df.iloc[i]["end"],int(df.iloc[i]["diff"])))
			label_new.extend([df.iloc[i]["Class"]] * int(df.iloc[i]["diff"]))
		
		point_new = [int(x) for x in point]
		df_new = pd.DataFrame({"file_id": doc, "Class": label_new, "point": point_new})
		df_new.drop_duplicates(inplace = True)

		return df_new

def change_continuous_non_text(df,step = 2):
	"""
	Convert the ref and hyp into discrete time-series (Time-based decision unit). The gap is set to 2 seconds by default
	e.g.(step = 1)
  	From
	docid	start end  valence             
	doc1	0     1     1                   
	doc1	1     2.1   2                  
	To
	docid	start end  valence
	doc1	0     1   (1*1)/1.0=1
	doc1	1     2   (2*1.0)/1.0=2
	"""
	start = list(df["start"])
	end = list(df["end"])
	label_list = list(df["Class"])
	time_pool = start + end
	if (max(time_pool) - min(time_pool)) % step == 0:
		step_list = np.linspace(min(time_pool), max(time_pool), int((max(time_pool) - min(time_pool))/step + 1))
	else:
		divisor = (max(time_pool) - min(time_pool)) // step
		remainder = (max(time_pool) - min(time_pool)) % step
		step_list = np.append(np.linspace(min(time_pool), max(time_pool) - remainder, int(divisor + 1)), max(time_pool))

	startpoint = []
	endpoint = []
	for i in range(len(step_list)-1):
			startpoint.append(round(step_list[i],3))
			endpoint.append(round(step_list[i+1],3))

	label_new_list = []
	for i,j in zip(startpoint,endpoint):
		label_new = 0
		for k in range(df.shape[0]):
			if i < end[k] and j > end[k] and i >= start[k]:
				if label_list[k] == silence_string:
					label_new = silence_string
				else:
					label_new = label_new + label_list[k]*(end[k]-i)

					while j > end[k]:
						k = k + 1
						if j > end[k]:
							if label_list[k] == silence_string:
								label_new = silence_string
							else:
								label_new = label_new + label_list[k]*(end[k]-start[k])

					if label_new != silence_string and label_list[k] != silence_string:
						label_new = label_new + label_list[k]*(j-start[k])
					else: 
						label_new = silence_string
						
			elif i < end[k] and j <= end[k] and i >= start[k]:


				if label_list[k] == silence_string:
					label_new = silence_string
				else:
					label_new = label_new + label_list[k]*(j-i)
			else:
				continue

		if label_new != silence_string:
			label_new = round(label_new/(j-i),3)
		label_new_list.append(label_new)
	
	df_new = pd.DataFrame({"start": startpoint, "end": endpoint, "Class": label_new_list})
	df_new["file_id"] = df.iloc[0]["file_id"]
	
	return df_new

def process_ref_hyp_time_series(ref, hyp, task):
	"""
	Convert the ref and hyp into discrete time-series that have the same length
	"""
	# Types = ref.loc[ref.Class != silence_string].type.unique()

	ref_dict = {}
	hyp_dict = {}
	Types = ref.loc[ref["Class"] != silence_string].type.unique()

	for type in Types:
		ref_dict[type] = {}
		hyp_dict[type] = {}

	segment_df = pd.DataFrame()
	file_ids = get_unique_items_in_array(ref['file_id'])
	for file_id in file_ids: 
	
		sub_ref = extract_df(ref, file_id)
		sub_hyp = extract_df(hyp, file_id)
		sub_type = list(sub_ref["type"])[0]

		# Check file type
		if list(sub_ref["type"])[0] == "text":
			continue_ref = change_continuous_text(sub_ref)
			
			pruned_continue_ref = continue_ref[continue_ref["Class"] != silence_string].copy()
			pruned_continue_ref.rename(columns={"Class": "continue_ref"}, inplace=True)

			if len(sub_hyp) != 0:
				continue_hyp = change_continuous_text(sub_hyp)
				# Get the time series of no speech in reference
				non_silence_point = list(continue_ref["point"][continue_ref["Class"] != silence_string])
				# Prune system using the time series of no speech in reference
				pruned_continue_hyp = continue_hyp[continue_hyp["point"].isin(non_silence_point)].copy()
				pruned_continue_hyp.rename(columns={"Class": "continue_hyp"}, inplace=True)

				ref_hyp_continue = pd.merge(pruned_continue_ref, pruned_continue_hyp)
				ref_dict[sub_type][file_id] = list(ref_hyp_continue["continue_ref"])
				hyp_dict[sub_type][file_id] = list(ref_hyp_continue["continue_hyp"])

				ref_hyp_continue["start"] = ref_hyp_continue["point"].astype(int)
				ref_hyp_continue["end"] = ref_hyp_continue["point"].astype(int)

				ref_hyp_continue = ref_hyp_continue[["start","end","continue_ref","file_id","continue_hyp"]]
			else:
				# If no output in hyp, add fake output 500 for valence and 1 for arousal
				ref_hyp_continue = pruned_continue_ref.copy()
				if task == "valence_continuous":
					ref_hyp_continue["continue_hyp"] = 500
				else:
					ref_hyp_continue["continue_hyp"] = 1

				ref_dict[sub_type][file_id] = list(ref_hyp_continue["continue_ref"])
				hyp_dict[sub_type][file_id] = list(ref_hyp_continue["continue_hyp"])

				ref_hyp_continue["start"] = ref_hyp_continue["point"].astype(int)
				ref_hyp_continue["end"] = ref_hyp_continue["point"].astype(int)

				ref_hyp_continue = ref_hyp_continue[["start","end","continue_ref","file_id","continue_hyp"]]
			segment_df = pd.concat([segment_df,ref_hyp_continue])

		else:
			continue_ref = change_continuous_non_text(sub_ref)

			pruned_continue_ref = continue_ref[continue_ref["Class"] != silence_string].copy()
			pruned_continue_ref.rename(columns={"Class": "continue_ref"}, inplace=True)

			if len(sub_hyp) != 0:
				continue_hyp = change_continuous_non_text(sub_hyp)
				# Get the time series of no speech in reference
				non_silence_df = continue_ref.loc[:,["start","end"]][continue_ref["Class"] != silence_string]
				# Prune system using the time series of no speech in reference
				pruned_continue_hyp = non_silence_df.merge(continue_hyp)
				pruned_continue_hyp.rename(columns={"Class": "continue_hyp"}, inplace=True)

				ref_hyp_continue = pd.merge(pruned_continue_ref, pruned_continue_hyp)
				ref_dict[sub_type][file_id] = list(ref_hyp_continue["continue_ref"])
				hyp_dict[sub_type][file_id] = list(ref_hyp_continue["continue_hyp"])

				ref_hyp_continue = ref_hyp_continue[["start","end","continue_ref","file_id","continue_hyp"]]
			else:
				# If no output in hyp, add fake output 500 for valence and 1 for arousal
				ref_hyp_continue = pruned_continue_ref.copy()
				if task == "valence_continuous":
					ref_hyp_continue["continue_hyp"] = 500
				else:
					ref_hyp_continue["continue_hyp"] = 1

				ref_dict[sub_type][file_id] = list(ref_hyp_continue["continue_ref"])
				hyp_dict[sub_type][file_id] = list(ref_hyp_continue["continue_hyp"])

				ref_hyp_continue = ref_hyp_continue[["start","end","continue_ref","file_id","continue_hyp"]]
			segment_df = pd.concat([segment_df,ref_hyp_continue])

	return ref_dict, hyp_dict, segment_df

def apply_level_label(list, task):
	"""
	Apply the level labels for both the reference and hypothesized system output
	positive → valence == 700-1000
	neutral → valence == 300-699
	negative → valence == 1-299

	high → arousal == 700-1000
	medium → arousal == 300-699
	low → arousal == 1-299
	"""

	labels = []
	if task == "valence_continuous":
		for i in list:
			if i >= 1 and i <= 299:
				label = "negative"
			if i >= 300 and i <= 699:
				label = "neutral"
			if i >= 700 and i <= 1000:
				label = "positive"
			labels.append(label)
	else:
		for i in list:
			if i >= 1 and i <= 299:
				label = "low"
			if i >= 300 and i <= 699:
				label = "medium"
			if i >= 700 and i <= 1000:
				label = "high"
			labels.append(label)

	return labels
			
def ccc(x,y):
  ''' Concordance Correlation Coefficient'''
  sxy = np.sum((x - np.mean(x))*(y - np.mean(y)))/len(x)
  rhoc = 2*sxy / (np.var(x) + np.var(y) + (np.mean(x) - np.mean(y))**2)
	
  return rhoc

def score_genre(ref_dict, hyp_dict):
	"""
	Generate CCC metric by genre
	"""
	result = {}
	ref = []
	hyp = []
	for genre in sorted(ref_dict.keys()):
		ref_genre = []
		hyp_genre = []
		for file_id in ref_dict[genre]:
			ref.extend(ref_dict[genre][file_id])
			hyp.extend(hyp_dict[genre][file_id])
			ref_genre.extend(ref_dict[genre][file_id])
			hyp_genre.extend(hyp_dict[genre][file_id])
		result[genre] = ccc(ref_genre, hyp_genre)
	result["all"] = ccc(ref, hyp)

	return result

def write_segment(segment_df, output_dir, task):
	"""
  Write segment diarization result into a file
  """
	if task == "valence_continuous":
		label = "valence"
	if task == "arousal_continuous":
		label = "arousal"
	
	segment_df["class"] = label
	segment_df_format = segment_df.copy()
	segment_df_format["ref"] = ["{:.3f}".format(x) for x in segment_df["continue_ref"]]
	segment_df_format["sys"] = ["{:.3f}".format(x) for x in segment_df["continue_hyp"]]
	segment_df_format["sort"] = [float(x) for  x in segment_df["start"]]

	segment_df_format["parameters"] = "{}"
	segment_df_format["start"] = [formatNumber(x) for x in segment_df["start"]]
	segment_df_format["end"] = [formatNumber(x) for x in segment_df["end"]]
	segment_df_format["window"] = "{start=" + segment_df_format["start"].astype(str) + ",end=" + segment_df_format["end"].astype(str) + "}"

	#segment_df_format = segment_df_format[["class","file_id","window","ref","sys","parameters"]]
	segment_df_sorted = segment_df_format.sort_values(by=['class', 'file_id', 'sort'])

	segment_df_sorted.to_csv(os.path.join(output_dir, "segment_diarization.tab"), index = False, quoting=3, sep="\t", escapechar="\t",
                                 columns=['class', 'file_id','window','ref','sys','parameters'])

def write_valence_arousal_scores(output_dir, CCC_result, task):
	"""
  Write aggregate result into a file
  """
	result_df = pd.DataFrame(columns=["task","genre","metric","value","correctness_criteria"])
	index = 0
	if task == "valence_continuous":
		label = "vd"
	if task == "arousal_continuous":
		label = "ad"
	for genre, value in CCC_result.items():
		result_df.loc[len(result_df.index)] = [label, genre, "CCC", round(value,3), "{}"]
		index = index + 1
	result_df_sorted = result_df.sort_values("genre")
	result_df_sorted.to_csv(os.path.join(output_dir, "scores_aggregated.tab"), sep = "\t", index = None)

def score_valence_arousal(ref, hyp, output_dir, task):
	"""
	The wrapper
	"""
	ref_dict, hyp_dict, segment_df = process_ref_hyp_time_series(ref, hyp, task)
	CCC_result = score_genre(ref_dict, hyp_dict)
	
	ensure_output_dir(output_dir)
	write_segment(segment_df, output_dir, task)
	write_valence_arousal_scores(output_dir, CCC_result, task)









	


