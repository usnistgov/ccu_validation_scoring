import os
import pandas as pd
import numpy as np
from .utils import *
from .preprocess_reference import *

def change_continuous_text(df):
	
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

  return df_new

def change_continuous_non_text(df,step = 2):

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
				if label_list[k] == "nospeech":
					label_new = "nospeech"
				else:
					label_new = label_new + label_list[k]*(end[k]-i)

					while j > end[k]:
						k = k + 1
						if j > end[k]:
							if label_list[k] == "nospeech":
								label_new = "nospeech"
							else:
								label_new = label_new + label_list[k]*(end[k]-start[k])

					if label_new != "nospeech" and label_list[k] != "nospeech":
						label_new = label_new + label_list[k]*(j-start[k])
					else: 
						label_new = "nospeech"
						
			elif i < end[k] and j <= end[k] and i >= start[k]:


				if label_list[k] == "nospeech":
					label_new = "nospeech"
				else:
					label_new = label_new + label_list[k]*(j-i)
			else:
				continue

		# print(i,j,label_new)
		if label_new != "nospeech":
			label_new = round(label_new/(j-i),3)
		label_new_list.append(label_new)
	
	df_new = pd.DataFrame({"start": startpoint, "end": endpoint, "Class": label_new_list})
	df_new["file_id"] = df.iloc[0]["file_id"]
	
	return df_new

def process_ref_hyp_time_series(ref, hyp):

	ref_list = []
	hyp_list = []

	file_ids = get_unique_items_in_array(ref['file_id'])

	for file_id in file_ids: 
		sub_ref = extract_df(ref, file_id)
		sub_hyp = extract_df(hyp, file_id)

		# Check file type
		if list(sub_ref["type"])[0] == "text":
			continue_ref = change_continuous_text(sub_ref)
			continue_hyp = change_continuous_text(sub_hyp)

			pruned_continue_ref = continue_ref[continue_ref["Class"] != "nospeech"].copy()
			pruned_continue_ref.rename(columns={"Class": "continue_ref"}, inplace=True)

			# Get the time series of no speech in reference
			non_silence_point = list(continue_ref["point"][continue_ref["Class"] != "nospeech"])
			# Prune system using the time series of no speech in reference
			pruned_continue_hyp = continue_hyp[continue_hyp["point"].isin(non_silence_point)].copy()
			pruned_continue_hyp.rename(columns={"Class": "continue_hyp"}, inplace=True)

		else:
			continue_ref = change_continuous_non_text(sub_ref)
			continue_hyp = change_continuous_non_text(sub_hyp)

			pruned_continue_ref = continue_ref[continue_ref["Class"] != "nospeech"].copy()
			pruned_continue_ref.rename(columns={"Class": "continue_ref"}, inplace=True)

			# Get the time series of no speech in reference
			non_silence_df = continue_ref.loc[:,["start","end"]][continue_ref["Class"] != "nospeech"]
			# Prune system using the time series of no speech in reference
			pruned_continue_hyp = non_silence_df.merge(continue_hyp)
			pruned_continue_hyp.rename(columns={"Class": "continue_hyp"}, inplace=True)

		ref_hyp_continue = pd.merge(pruned_continue_ref, pruned_continue_hyp)
		ref_list.extend(list(ref_hyp_continue["continue_ref"]))
		hyp_list.extend(list(ref_hyp_continue["continue_hyp"]))

		return(ref_list, hyp_list)

def ccc(x,y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - np.mean(x))*(y - np.mean(y)))/len(x)
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (np.mean(x) - np.mean(y))**2)

    return rhoc

def write_valence_arousal_scores(output_dir, CCC_result):

	result_df = pd.DataFrame({"Metric": ["CCC"], "Score": [CCC_result]})
	result_df.to_csv(os.path.join(output_dir, "system_scores.csv"), index = None)

def score_valence_arousal(ref, hyp, output_dir):

	ref_list, hyp_list = process_ref_hyp_time_series(ref, hyp)
	CCC_result = ccc(ref_list, hyp_list)
	write_valence_arousal_scores(output_dir, CCC_result)









	


