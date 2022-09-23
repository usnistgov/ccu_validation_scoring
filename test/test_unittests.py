from heapq import merge
from lib2to3.pgen2.pgen import generate_grammar
from CCU_validation_scoring.preprocess_reference import *
import json
import ast
import unittest
import pandas as pd

class NormDiscoveryTests(unittest.TestCase):
    #  For norm discovery, since there is only one annotation pass,
    # only "Instance Merging" is performed

    def setUp(self):
        current_path = os.path.abspath(__file__)
        test_dir_path = os.path.dirname(current_path)

        self.merge_files_dir = os.path.join(test_dir_path, 'scores/unittests/merge_test_files')

    def test_merge_vote_time_periods(self): # /Users/cnc30/Documents/VisualStudioProjects/ccu_validation_scoring/test/scores/unittests/mergeInputFile1.txt
        # Test Case: merge_vote_time_periods correctly merges time periods if gap is appropriate

        merge_test_files = os.listdir(self.merge_files_dir)
        num_input_files = int(len(merge_test_files) / 2)

        file_counter = 1
        while file_counter <= num_input_files:
        
            # Find input file i and corresponding expected output file
            input_index = merge_test_files.index("mergeInputFile" + str(file_counter) + ".txt")
            expected_index = merge_test_files.index("mergeExpectedFile" + str(file_counter) + ".txt")
            input_file_path = os.path.join(self.merge_files_dir, merge_test_files[input_index])
            expected_file_path = os.path.join(self.merge_files_dir, merge_test_files[expected_index])

            # Create an input dict
            with open(input_file_path) as file: 
                data = file.read()
            input_dict = json.loads(data)

            # Create a dictionary that is returned by merge_vote_time_periods
            generated_list = merge_vote_time_periods(input_dict)
                
            # Create expected list
            with open(expected_file_path) as file:
                data = file.read()
            expected_list = json.loads(data)
                
            # Compare generated dict to the expected dict
            for i  in range(0, len(generated_list)):
                is_accurate = generated_list[i] == expected_list[i]
                if not is_accurate:
                    self.assertEqual(generated_list[i]['range'], expected_list[i]['range'], msg=str(generated_list[i]['range']) + " does not equal " + str(expected_list[i]['range']))
                    self.assertEqual(generated_list[i]['Class'], expected_list[i]['Class'], msg=str(generated_list[i]['Class']) + " does not equal " + str(expected_list[i]['Class']))

            file_counter += 1


class EmotionDetectionTests(unittest.TestCase):
    # For emotion detection, since there are more than one annotation passes (up to 3),
    # "Judgment Collapsing by Majority Voting" will be applied, followed by "Instance Merging"

    def setUp(self):
        current_path = os.path.abspath(__file__)
        test_dir_path = os.path.dirname(current_path)

        self.vote_files_dir = os.path.join(test_dir_path, 'scores/unittests/vote_test_files')

    def test_get_highest_vote_input_processing(self, input_file_path='scores/unittests/vote_test_files/voteInputFile1.csv', expected_file_path='scores/unittests/vote_test_files/voteExpectedFile1.csv'):
        # Test Case: Tests that get_highest_vote_based_on_time reads in data frame correctly,
        # & returns valid dict w/ expected keys and values

        vote_test_files = os.listdir(self.vote_files_dir)
        num_input_files = int(len(vote_test_files) / 2)

        file_counter = 1
        while file_counter <= num_input_files: 
           
            input_index = vote_test_files.index("voteInputFile" + str(file_counter) + ".csv")
            expected_index = vote_test_files.index("voteExpectedFile" + str(file_counter) + ".csv")
            input_file_path = os.path.join(self.vote_files_dir, vote_test_files[input_index])
            expected_file_path = os.path.join(self.vote_files_dir, vote_test_files[expected_index])
            
             # Create input data frame
            input_df = pd.read_csv(input_file_path)
            input_df.user_id = input_df.user_id.astype(str)
            input_df.multi_speaker = input_df.multi_speaker.astype(str) # str or bool?
            input_df.start = input_df.start.astype(str)
            input_df.end = input_df.end.astype(str)
            input_df.Class = input_df.Class.astype(str)
      
            # Call get_highest_vote_based_on_time w/ input data frame
            emo_dict = get_highest_vote_based_on_time(input_df)

            # Test that emo_dict is a dict w/ strings as keys & list as values
            self.assertIsInstance(emo_dict, dict)        
        
            for key in emo_dict.keys():
                self.assertIsInstance(key, str)
                
            for value in emo_dict.values():
                self.assertIsInstance(value, list)

            # Create expected data frame
            expected_df = pd.read_csv(expected_file_path)

            # Loop over each column and change strings to dicts
            for (column_name, column_data) in expected_df.iteritems():
                if column_name != "Emotion":
                    # Convert data to a dict
                    for i in range(0, len(expected_df[column_name])):
                        expected_df[column_name][i] = ast.literal_eval(expected_df[column_name][i])

            # Create data frame generated by get_highest_vote dict
            generated_df = pd.DataFrame.from_dict(emo_dict, orient='index')
            generated_df = generated_df.reset_index()
            generated_df = generated_df.rename(columns={"index": "Emotion"})

            # Convert column names from ints to strings
            for i in range(0, len(generated_df.columns.values)):
                generated_df.columns.values[i] = str(generated_df.columns.values[i])

            # Compare generated dataframe values to expected dataframe values
            for i in range(0, len(generated_df.columns.values)):
                column_name = generated_df.columns.values[i]
                for j in range(0, len(generated_df.index)):
                    if column_name != "Emotion":
                        msg = "time dict is inaccurate"
                    else:
                        msg = "Emotion is inaccurate"
                    self.assertEqual(generated_df[column_name][j], expected_df[column_name][j], msg)

            file_counter += 1

class ArousalAndValenceTests(unittest.TestCase):
    # For arousal and valence detection, since there are more than one annotation passes (up tp 3).
    # "Judgment Averaging" will be applied, followed by converting it into time series

    def setUp(self):
        current_path = os.path.abspath(__file__)
        test_dir_path = os.path.dirname(current_path)

        self.avg_files_dir = os.path.join(test_dir_path, 'scores/unittests/avg_test_files')

    def test_get_average_score_input_processing(self):
        # Test Case: Tests whether or not get_average_score_based_on_time reads in data_frame correctly, 
        # & returns valid dict w/ appropriate keys and values

        avg_test_files = os.listdir(self.avg_files_dir)
        num_input_files = int(len(avg_test_files) / 2)

        file_counter = 1
        while file_counter <= num_input_files:
            
            input_index = avg_test_files.index("avgInputFile" + str(file_counter) + ".csv")
            expected_index = avg_test_files.index("avgExpectedFile" + str(file_counter) + ".csv")
            input_file_path = os.path.join(self.avg_files_dir, avg_test_files[input_index])
            expected_file_path = os.path.join(self.avg_files_dir, avg_test_files[expected_index])

            # Create input data frame
            input_df = pd.read_csv(input_file_path)
            input_df.user_id = input_df.user_id.astype(str)
            input_df.file_id = input_df.file_id.astype(str)
            input_df.segment_id = input_df.segment_id.astype(str)
            input_df.start = input_df.start.astype(str)
            input_df.end = input_df.end.astype(str)
            input_df.Class = input_df.Class.astype(str)


            # Call get_average_score_based_on_time w/ above data frame
            time_dict = get_average_score_based_on_time(input_df)

            # Test that time_dict is a dict w/ strings as keys & dictionaries as values
            self.assertIsInstance(time_dict, dict)        
        
            for key in time_dict.keys():
                self.assertIsInstance(key, str)
                
            for value in time_dict.values():
                self.assertIsInstance(value, dict)

            # Create expected data frame from CSV
            expected_df = pd.read_csv(expected_file_path)

            # Clean the data
            expected_df.start = expected_df.start.astype(str)
            expected_df.end = expected_df.end.astype(str)
            expected_df.user_id = expected_df.user_id.astype(str)

            for key in expected_df.columns.values:
                if type(expected_df[key]) is str:
                    expected_df[key] = expected_df[key].str.strip()

            # Create generated data frame to compare to expected data frame
            generated_df = pd.DataFrame.from_dict(time_dict, orient='index')
            generated_df = generated_df.reset_index()
            generated_df = generated_df.rename(columns={"index": "key"})
            generated_df.set_index(pd.Index([0,1,2]),drop=True)

            # Test that generated df contains correct columns (key, file id, segment_id, start, end, user_id, class)
            expected_column_names = ['key', 'file_id', 'segment_id', 'start', 'end', 'user_id', 'Class']
            generated_column_names = generated_df.columns.values
        
            column_counter = 0
            while column_counter < len(expected_column_names):
                self.assertEqual(generated_column_names[column_counter], expected_column_names[column_counter], msg=generated_column_names[column_counter] + " column not found.")
                column_counter = column_counter + 1

            key = 'key'
            file_id = 'file_id'
            segment_id = 'segment_id'
            start = 'start'
            end = 'end'
            user_id = 'user_id'
            class_var = 'Class'
            counter = 0

            while counter < len(generated_df.index):
                self.assertAlmostEqual(generated_df[key][counter], expected_df[key][counter])
                self.assertAlmostEqual(generated_df[file_id][counter], expected_df[file_id][counter])
                self.assertAlmostEqual(generated_df[segment_id][counter], expected_df[segment_id][counter])
                self.assertAlmostEqual(generated_df[start][counter], expected_df[start][counter])
                self.assertAlmostEqual(generated_df[end][counter], expected_df[end][counter])
                self.assertAlmostEqual(generated_df[user_id][counter], expected_df[user_id][counter])
                self.assertAlmostEqual(generated_df[class_var][counter], expected_df[class_var][counter])
                counter = counter + 1

            file_counter += 1

if __name__ == '__main__':
    unittest.main() 