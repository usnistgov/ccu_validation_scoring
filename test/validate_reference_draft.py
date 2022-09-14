import unittest
import logging
from reference.LDC_reference_sample.data import *
from os.path import exists
import pandas as pd

logger = logging.getLogger('VALIDATION')

# File Paths
norms_path = "reference/LDC_reference_sample/data/norms.tab" #test/reference/LDC_reference_sample/data/norms.tab
emotions_path = "reference/LDC_reference_sample/data/emotions.tab"
changepoint_path = "reference/LDC_reference_sample/data/changepoint.tab"
valence_arousal_path = "reference/LDC_reference_sample/data/valence_arousal.tab"
segments_path = "reference/LDC_reference_sample/docs/segments.tab"
system_input_path = "reference/LDC_reference_sample/docs/system_input.index.tab"
file_list = [norms_path, emotions_path, changepoint_path, valence_arousal_path, segments_path, system_input_path]

class UnitTests(unittest.TestCase):
	
	def test_norm_range(self, file='reference/LDC_reference_sample/data/norms.tab'):
		
		df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
		valid_norms = ['001', '002', '007', 'none', 'nospeech'] #subject to change	
		if df.shape[0] != 0:
			invalid_norms = []
			for norm in df['norm']:
				if norm not in valid_norms:
					invalid_norms.append(norm)

			if len(invalid_norms) > 0:
				logger.error("Invalid file {}", file)
				logger.error("Additional emotion(s) '{}' have been found in {} ".format(set(invalid_norms), file))
				return False
		
		return True
	
	def test_nospeech_all_columns(self, file=valence_arousal_path):
		
		df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
		for index, row in df.iterrows():
			val_and_arousal_tags = [row['valence_continuous'], row['valence_binned'], row['arousal_continuous'], row['arousal_binned']]
			if "nospeech" in val_and_arousal_tags:
				nospeech_count = val_and_arousal_tags.count("nospeech")
				if nospeech_count != 4:
					logger.error("Validation failed")
					logger.error("Expected 4 'nospeech' tags but only got {} in file {}".format(nospeech_count, file))
					
	def test_nospeech_all_annotators(self, file="reference/LDC_reference_sample/data/valence_arousal.tab"):
		
		df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
		# Make sure "nospeech" tags applied to all value columns
		UnitTests.test_nospeech_all_columns(file)
		
		last_segment_id = ""
		for index, row in df.iterrows():
			# Create a partial dataframe containing only rows w/ this specific segment_id
			current_segment_id = row['segment_id']
			if current_segment_id != last_segment_id:
				partial_df = df[df['segment_id'] == current_segment_id]
			else:
				continue

			# Check if this segment_id has any "nospeech values". If so, make sure that all annotators said no speech
			nospeech_df = partial_df[partial_df['valence_continuous'] == "nospeech"]
			if nospeech_df.shape[0] == 0:
				last_segment_id = current_segment_id
				continue
			else: 
				# Retrieve user_ids of those who did not tag nospeech
				missing_nospeech_df = partial_df[partial_df['valence_continuous'] != "nospeech"]
				if missing_nospeech_df.shape[0] != 0:
					logger.error("Validation failed")
					logger.error("Inconsistent 'nospeech' tags in segment {}".format(current_segment_id))
					logger.error("The following annotators tagged 'nospeech': {}".format(set(nospeech_df['user_id'])))
					logger.error("The following annotators did not tag 'nospeech': {}".format(set(missing_nospeech_df['user_id'])))

			last_segment_id = current_segment_id

	def test_empty_na(self, task="norms", file=norms_path):
		
		df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
		
		for index, row in df.iterrows():
			# Check second to last row (row['emotion'] for emotions.tab and row['norm'])
			if task == "emotions":
				if row["emotion"] == "none" and row['multi_speaker'] != "EMPTY_NA":
					logger.error("Validation failed {}".format(file))
					logger.error("Expected 'EMPTY_NA' status, got {}".format(row['multi_speaker']))
			if task == "norms":
				if row["norm"] == "none" and row['status'] != "EMPTY_NA":
					logger.error("Validation failed {}".format(file))
					logger.error("Expected 'EMPTY_NA' status, got {}".format(row['status']))
	
	def test_duplicate_emotions(self, file=emotions_path):
		
		df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
		for index, row in df.iterrows():
			emotion_list = row['emotion'].split()
			uniq_emotions = set(emotion_list)
			if len(emotion_list) != len(uniq_emotions):
				logger.error("Validation failed")
				seen = set()
				duplicates = [emotion for emotion in emotion_list if emotion in seen or seen.add(emotion)]
				if duplicates:
					logger.error("File {} contains the following emotion(s) duplicated: {}".format(file, duplicates))

	def test_start_end_types(self, file=segments_path):
	
		input_df = pd.read_csv(system_input_path, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

		text_ids = []
		audvid_ids = []
		for index, row in input_df.iterrows():
			if row['type'] == "text":
				text_ids.append(row['file_id'])
			elif row['type'] == "audio" or row['type'] == "video":
				audvid_ids.append(row['file_id'])
			else:
				logger.error("Invalid file")
				logger.error("File {} has a type other than text, audio, or video for file_id {}".format(system_input_path), row['file_id'])
		
		# Open segments.tab and for every file_id, if it's in one list, make sure it has a data type of ___ and vice versa otherwise
		segments_df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
		for index, row in segments_df.iterrows():
			if row['file_id'] in text_ids:
				if row['start'].is_integer() == row['end'].is_integer() == False:
					print("Start: {}, End: {}".format(row['start'], row['end']))
					logger.error("Validation failed")
					logger.error("In file {}, file id {} contains start/end time that are not ints".format(file, row['file_id']))
			elif row['file_id'] in audvid_ids:
				if isinstance(row['start'], float) == isinstance(row['end'], float) == False:
					logger.error("Validation failed")
					logger.error("In file {}, file id {} contains start/end time that are not floats".format(file, row['file_id']))
			else:
				logger.error("Validation failed")
				logger.error("Type of file ID {} in file {} is not text, audio, or video".format(row['file_id'], file))

	def test_fileid_segmentid_match(self, file=emotions_path):
	
		segments_df = pd.read_csv(segments_path, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')

		expected_file_ids = []
		expected_segment_ids = []
		for index, row in segments_df.iterrows():
			if row['file_id'] not in expected_file_ids:
				expected_file_ids.append(row['file_id'])
			if row['segment_id'] not in expected_segment_ids:
				expected_segment_ids.append(row['segment_id'])
		
		# For each of the other 3 files, loop through and make sure that the file and segment ids are in those lists
		invalid_file_ids = []
		invalid_segment_ids = []
		df = pd.read_csv(file, dtype={'norm': object, 'sys_norm': object, 'ref_norm': object, 'message': object}, sep='\t')
		if df.shape[0] != 0:
			for index, row in df.iterrows():
				if row['file_id'] not in expected_file_ids:
					invalid_file_ids.append(row['file_id'])
				if row['segment_id'] not in expected_segment_ids:
					invalid_segment_ids.append(row['segment_id'])
		
			if len(invalid_file_ids) > 0 or len(invalid_segment_ids) > 0:
				logger.error("Invalid file {}".format(file))
				if len(invalid_file_ids) > 0:
					logger.error("File ID(s) {} in {} is not found in {}".format(invalid_file_ids, file, segments_path))
				if len(invalid_segment_ids) > 0:
					logger.error("Segment ID(s) {} in {} is not found in {}".format(invalid_segment_ids, file, segments_path))
	


if __name__ == '__main__':
	unittest.main()

