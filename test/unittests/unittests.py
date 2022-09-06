from CCU_validation_scoring.preprocess_reference import *
import json
import ast
import sys
import unittest
import pandas as pd

class NormDiscoveryTests(unittest.TestCase):
    #  For norm discovery, since there is only one annotation pass,
    # only "Instance Merging" is performed
    def test_merge_vote_time_periods(self,input_file_path="mergeInputFile1.txt", expected_path_file="mergeExpectedFile1.txt"):
    # Test Case: merge_vote_time_periods correctly merges time periods if gap is appropriate
        #data = ""
        # Create an input dict
        with open(input_file_path) as file: 
            data = file.read()
        input_dict = json.loads(data)

        # Create expected list
        with open(expected_path_file) as file:
            data = file.read()
        expected_list = json.loads(data)

        # Create a dictionary that is returned by merge_vote_time_periods
        generated_list = merge_vote_time_periods(input_dict)   

        # Compare generated dict to the expected dict
        for i  in range(0, len(generated_list)):
            is_accurate = generated_list[i] == expected_list[i]
            if not is_accurate:
                self.assertEqual(generated_list[i]['range'], expected_list[i]['range'], msg=str(generated_list[i]['range']) + " does not equal " + str(expected_list[i]['range']))
                self.assertEqual(generated_list[i]['Class'], expected_list[i]['Class'], msg=str(generated_list[i]['Class']) + " does not equal " + str(expected_list[i]['Class']))



class EmotionDetectionTests(unittest.TestCase):
    # For emotion detection, since there are more than one annotation passes (up to 3),
    # "Judgment Collapsing by Majority Voting" will be applied, followed by "Instance Merging"

    def test_get_highest_vote_input_processing(self, input_file_path='inputeFile1GetHighestVote.csv', expected_file_path='expectedFile1GetHighestVote.csv'):
        # Test Case: Tests that get_highest_vote_based_on_time reads in data frame correctly,
        # & returns valid dict w/ expected keys and values

        # Create input data frame
        input_df = pd.read_csv(input_file_path)
        input_df.user_id = input_df.user_id.astype(str)
        #input_df.file_id = input_df.file_id.astype(str)
        #input_df.segment_id = input_df.segment_id.astype(str)
        input_df.multi_speaker = input_df.multi_speaker.astype(str) # str or bool?
        input_df.start = input_df.start.astype(str)
        input_df.end = input_df.end.astype(str)
        #input_df.type = input_df.type.astype(str)
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

        # # Convert column names from strings to ints
        # for i in range(1, len(expected_df.columns.values)):
        #     expected_df.columns.values[i] = eval(expected_df.columns.values[i])

        # Loop over each column and change strings to dicts
        for (column_name, column_data) in expected_df.iteritems():
            if column_name != "Emotion":
                # Convert data to a dict
                for i in range(0, len(expected_df[column_name])):
                    expected_df[column_name][i] = ast.literal_eval(expected_df[column_name][i])
                    # print(expected_df[column_name][i])
                    # print(type(expected_df[column_name][i]))

        # Create data frame generated by get_highest_vote dict
        generated_df = pd.DataFrame.from_dict(emo_dict, orient='index')
        generated_df = generated_df.reset_index()
        generated_df = generated_df.rename(columns={"index": "Emotion"})

        # Convert column names from ints to strings
        for i in range(0, len(generated_df.columns.values)):
            generated_df.columns.values[i] = str(generated_df.columns.values[i])

        # Testing
        for i in range(0, len(generated_df.columns.values)):
            column_name = generated_df.columns.values[i]
            for j in range(0, len(generated_df.index)):
                if column_name != "Emotion":
                    msg = "time dict is inaccurate"
                else:
                    msg = "Emotion is inaccurate"
                self.assertEqual(generated_df[column_name][j], expected_df[column_name][j], msg)

class ArousalAndValenceTests(unittest.TestCase):
    # For arousal and valence detection, since there are more than one annotation passes (up tp 3).
    # "Judgment Averaging" will be applied, followed by converting it into time series

    def test_get_average_score_input_processing(self, input_file_path='inputFile1GetAvgScore.csv', expected_file_path='expectedFile1GetAvgScore.csv'):
        # Take in name of two files
        # Test Case: Tests whether or not get_average_score_based_on_time reads in data_frame correctly, 
        # & returns valid dict w/ appropriate keys and values

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
            # Test edge case (just one person, "nospeech")
            # Create input, output, and generated data frame and then compare
            # Handle multiple testing data setups

        # # Printing for testing
        # print(generated_df)
        # print(expected_df)
        key = 'key'
        file_id = 'file_id'
        segment_id = 'segment_id'
        start = 'start'
        end = 'end'
        user_id = 'user_id'
        class_var = 'Class'
        counter = 0

        while counter < len(generated_df.index):
            self.assertEqual(generated_df[key][counter], expected_df[key][counter])
            self.assertEqual(generated_df[file_id][counter], expected_df[file_id][counter])
            self.assertEqual(generated_df[segment_id][counter], expected_df[segment_id][counter])
            self.assertEqual(generated_df[start][counter], expected_df[start][counter])
            self.assertEqual(generated_df[end][counter], expected_df[end][counter])
            self.assertEqual(generated_df[user_id][counter], expected_df[user_id][counter])
            self.assertEqual(generated_df[class_var][counter], expected_df[class_var][counter])
            counter = counter + 1


    # def test_get_average_score_average_calculation(self):
    #     # Test Case: Tests whether get_average_score_based_on_time calculates accurate and precisely rounded averages
       
    #     # Create data frame
    #     data = [
    #         ['513', 'M010029SP', 'M010029SP_0001', 5, 288 , 2, 200.133, 215.133, 'video', 874], 
    #         ['98543', 'M010029SP',  'M010029SP_0001', 4, 107, 1, 200.133, 215.133, 'video', 799], 
    #         ['2297', 'M010015BY','M010015BY_0001', 4, 288, 2, 1104.567, 1119.567, 'video', 669],
    #         ['513', 'M010015BY', 'M010015BY_0001', 3, 478, 3, 1104.567, 1119.567,'video', 503],
    #         ['8765', 'M010005QD','M010005QD_0001', 2, 666, 4, 45.2, 60.2, 'video', 234],
    #         ['44318', 'M010005QD', 'M010005QD_0001', 3, 503, 3, 45.2, 60.2, 'video', 433]
    #     ]
    #     data_frame = pd.DataFrame(data, columns = ['user_id', 'file_id', 'segment_id', 'valence_binned', 
    #                                                 'arousal_continuous', 'arousal_binned', 'start', 'end', 'type', 'Class'])

    #     # Call get_average_score_based_on_time w/ above data frame
    #     time_dict = get_average_score_based_on_time(data_frame)

    #     # Test that averages are accurate
    #     time_key1 = "200.133 - 215.133"
    #     time_key2 = '1104.567 - 1119.567'
    #     time_key3 = '45.2 - 60.2'
    #     expected_average1 = 836.5
    #     expected_average2 = 586
    #     expected_average3 = 333.5

    #     self.assertEqual(time_dict[time_key1]['Class'], expected_average1, msg="Incorrect average calculation") 
    #     self.assertEqual(time_dict[time_key2]['Class'], expected_average2, msg="Incorrect average calculation")
    #     self.assertEqual(time_dict[time_key3]['Class'], expected_average3, msg="Incorrect average calculation")
        


    # def test_get_average_score_valence_arousal_within_range(self): 
    #     # Already validated, not needed?
    #     # Test Case: 'Class' keys of embedded dicts returned by get_average_score_based_on_time contain floats between 1 and 1000
    #     # or a valid string (e.g. "nospeech")
    #     # Create data frame, pass to fcn, test computations, validate results 

    #      # Create data frame
    #     data = [
    #         ['513', 'M010029SP', 'M010029SP_0001', 5, 288 , 2, 200.133, 215.133, 'video', 874], 
    #         ['98543', 'M010029SP',  'M010029SP_0001', 4, 107, 1, 200.133, 215.133, 'video', 799], 
    #         ['2297', 'M010015BY','M010015BY_0001', 4, 288, 2, 1104.567, 1119.567, 'video', 669],
    #         ['513', 'M010015BY', 'M010015BY_0001', 3, 478, 3, 1104.567, 1119.567,'video', 503],
    #         ['8765', 'M010005QD','M010005QD_0001', 2, 666, 4, 45.2, 60.2, 'video', 234],
    #         ['44318', 'M010005QD', 'M010005QD_0001', 3, 503, 3, 45.2, 60.2, 'video', 433]
    #     ]
    #     data_frame = pd.DataFrame(data, columns = ['user_id', 'file_id', 'segment_id', 'valence_binned', 
    #                                                 'arousal_continuous', 'arousal_binned', 'start', 'end', 'type', 'Class'])

    #     # Call get_average_score_based_on_time w/ above data frame
    #     time_dict = get_average_score_based_on_time(data_frame)

    #     # Test that averages are within range
    #     min_score = 1        # minimum valence/arousal value
    #     max_score = 1000     # maximum valence/arousal value
        
    #     for value in time_dict.values():
    #         average = value['Class']
    #         self.assertIsInstance(average, float)
    #         self.assertGreaterEqual(average, min_score)
    #         self.assertLessEqual(average, max_score)

    #     # Should I test that an exception is throw if value is out of range?
    


if __name__ == '__main__':
    # Consider running program from command line w/ -v for more thorough output
    # test_obj = ArousalAndValenceTests()
    # test_obj.test_get_average_score_input_processing()
    # test_obj.test_get_average_score_average_calculation()
    # test_obj.test_get_average_score_valence_arousal_within_range()
    # input_file_path = 'inputFile1GetAvgScore.csv'
    # expected_file_path = 'expectedFile1GetAvgScore.csv'
    unittest.main() 