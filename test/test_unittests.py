from CCU_validation_scoring.preprocess_reference import *
from CCU_validation_scoring.score_norm_emotion import *
import json
import ast
import unittest
import pandas as pd


class IOUTests_v1(unittest.TestCase):
    #  To test the IoU
    # 

    def setUp(self):
        self.cases = [ { 'expected': [ [0.4], [30], [75], [55], [0.73333] ] ,
                         'inputs': { 'ref': pd.DataFrame(data={'ref_start': [ 10 ], 'ref_end': [ 40 ]}) , 'tgts': [ 0.0, 75 ] } },
                       { 'expected': [ [0.4], [30], [75], [55], [0.73333] ] ,
                         'inputs': { 'ref': pd.DataFrame(data={'ref_start': [  0 ], 'ref_end': [ 75 ]}) , 'tgts': [ 10,  40] } },
                       { 'expected': [ [0.8125], [65], [80], [80], [1.0] ] ,
                         'inputs': { 'ref': pd.DataFrame(data={'ref_start': [ 10 ], 'ref_end': [ 80 ]}) , 'tgts': [ 0.0, 75 ] } },
                       { 'expected': [ [0.8125], [65], [80], [80], [1.0] ] ,
                         'inputs': { 'ref': pd.DataFrame(data={'ref_start': [  0 ], 'ref_end': [ 75 ]}) , 'tgts': [ 10,  80 ] } }, 
                       { 'expected': [ [0.98], [49], [50], [50], [1.0] ] ,
                         'inputs': { 'ref': pd.DataFrame(data={'ref_start': [  1 ], 'ref_end': [ 50 ]}) , 'tgts': [ 0,  50 ] } }]
        
    def test_iou(self):
        for case in range(len(self.cases)):
            ret = segment_iou_v1(self.cases[case]['inputs']['ref']['ref_start'],
                                 self.cases[case]['inputs']['ref']['ref_end'],
                                 self.cases[case]['inputs']['tgts'])
            [ self.assertAlmostEqual(exp, calc, 3, msg=f"IoU Check Fails: case={case} exp={exp} != calc={calc}")             for exp, calc in zip(self.cases[case]['expected'][0], ret[0]) ]
            [ self.assertAlmostEqual(exp, calc, 3, msg=f"intersection Check Fails: case={case} exp={exp} != calc={calc}")    for exp, calc in zip(self.cases[case]['expected'][1], ret[1]) ]
            [ self.assertAlmostEqual(exp, calc, 3, msg=f"union Check Fails: case={case} exp={exp} != calc={calc}")           for exp, calc in zip(self.cases[case]['expected'][2], ret[2]) ]
            [ self.assertAlmostEqual(exp, calc, 3, msg=f"cb_intersection Check Fails: case={case} exp={exp} != calc={calc}") for exp, calc in zip(self.cases[case]['expected'][3], ret[3]) ]
            [ self.assertAlmostEqual(exp, calc, 3, msg=f"cb_IoU Check Fails: case={case} exp={exp} != calc={calc}")          for exp, calc in zip(self.cases[case]['expected'][4], ret[4]) ]


# class IOUTests_v2_save(unittest.TestCase):
#     #  To test the IoU
#     # 

#     def setUp(self):
#         ##############################  IoU     Inter Union   csb   cse   ScTP      ScFP
#         #self.cases = [ { 'expected': [ [0.75, 0.75],   [30, 30], [40, 40], [20, 20], [60, 60],  [1.0, 1.0],    [0.333, 0.333] ] ,  'inputs': { 'ref': pd.DataFrame(data={'ref_start': [ 25 ], 'ref_end': [ 55 ]}) , 'tgts': [ [ 20, 60 ], [ 20, 60] ] } }, 
#         self.cases = [ { 'expected': [ [0.75],   [30], [40], [20], [60],  [1.0],    [0.333] ] ,  'inputs': { 'file': 'F4', 'ref': pd.DataFrame(data={'ref_start': [ 25 ], 'ref_end': [ 55 ]}) , 'tgts': [ [20, 60] ] } }, 
#                        { 'expected': [ [0.75],   [30], [40], [20], [60],  [1.0],    [0.0] ] ,    'inputs': { 'file': 'F5', 'ref': pd.DataFrame(data={'ref_start': [ 20 ], 'ref_end': [ 60 ]}) , 'tgts': [ [25, 55] ] } },
#                        { 'expected': [ [0.4286], [30], [70],  [5], [75],  [1.0],    [1.333] ] ,  'inputs': { 'file': 'F6', 'ref': pd.DataFrame(data={'ref_start': [ 25 ], 'ref_end': [ 55 ]}) , 'tgts': [ [ 5, 75] ] } },
#                        { 'expected': [ [0.4286], [30], [70], [10], [70],  [0.8571], [0.0] ] ,    'inputs': { 'file': 'F7', 'ref': pd.DataFrame(data={'ref_start': [  5 ], 'ref_end': [ 75 ]}) , 'tgts': [ [25, 55] ] } },
#                        { 'expected': [ [0.5454], [30], [55], [20], [75],  [1.0],    [0.5714] ] , 'inputs': { 'file': 'F8', 'ref': pd.DataFrame(data={'ref_start': [ 20 ], 'ref_end': [ 55 ]}) , 'tgts': [ [25, 75] ] } },
#                        { 'expected': [ [0.50],   [30], [60], [10], [65],  [0.9],    [0.2] ] ,    'inputs': { 'file': 'F9', 'ref': pd.DataFrame(data={'ref_start': [  5 ], 'ref_end': [ 55 ]}) , 'tgts': [ [25, 65] ] } },
#                        { 'expected': [ [0.5454], [30], [55], [20], [70],  [0.9],    [0.1] ] ,    'inputs': { 'file': 'F10', 'ref': pd.DataFrame(data={'ref_start': [ 25 ], 'ref_end': [ 75 ]}) , 'tgts': [ [20, 55] ] } },
#                        { 'expected': [ [0.1667], [10], [60],  [5], [50],  [0.625],  [0.5] ] ,    'inputs': { 'file': 'F11', 'ref': pd.DataFrame(data={'ref_start': [ 25 ], 'ref_end': [ 65 ]}) , 'tgts': [ [ 5, 35] ] } }
#                       ]
                                                
#     def test_iou(self):
#         fig, ax = plt.subplots(len(self.cases), 1, figsize=(8,1.5 * len(self.cases)), constrained_layout=True)

#         for case in range(len(self.cases)):
#             #print(self.cases[case])
#             #print(self.cases[case]['inputs']['tgts'])
                      
#             #d = pd.DataFrame([[self.cases[case]['inputs']['tgts'][0], self.cases[case]['inputs']['tgts'][1] ]], columns = ['start', 'end'])
#             d = pd.DataFrame(self.cases[case]['inputs']['tgts'], columns = ['start', 'end'])
#             #print("")
#             #print(f"df wtf type {type(d)}")
#             #print(d)
#             ret = segment_iou_v2(self.cases[case]['inputs']['ref']['ref_start'][0],
#                                  self.cases[case]['inputs']['ref']['ref_end'][0],
#                                  [ d.start, d.end],
#                                  15)
            
#             print(f"ref_start {self.cases[case]['inputs']['ref']['ref_start'][0]} ref_end {self.cases[case]['inputs']['ref']['ref_end'][0]} -> tgts {self.cases[case]['inputs']['tgts']}")
#             print("IoU  [0]: " + str(ret[0].to_list()))
#             print("Int  [1]: " + str(ret[1].to_list()))
#             print("Uni  [2]: " + str(ret[2].to_list()))
#             print("csb  [3]: " + str(ret[3].to_list()))
#             print("cse  [4]: " + str(ret[4].to_list()))
#             print("ScTP [5]: " + str(ret[5].to_list()))
#             print("ScFP [6]: " + str(ret[6].to_list()))
#             exit(0)
            
#             [ self.assertAlmostEqual(exp, calc, 3, msg=f"IoU Check Fails: case={case} exp={exp} != calc={calc}")             for exp, calc in zip(self.cases[case]['expected'][0], ret[0]) ]
#             [ self.assertAlmostEqual(exp, calc, 3, msg=f"intersection Check Fails: case={case} exp={exp} != calc={calc}")    for exp, calc in zip(self.cases[case]['expected'][1], ret[1]) ]
#             [ self.assertAlmostEqual(exp, calc, 3, msg=f"union Check Fails: case={case} exp={exp} != calc={calc}")           for exp, calc in zip(self.cases[case]['expected'][2], ret[2]) ]
#             [ self.assertAlmostEqual(exp, calc, 3, msg=f"csb Check Fails: case={case} exp={exp} != calc={calc}")             for exp, calc in zip(self.cases[case]['expected'][3], ret[3]) ]
#             [ self.assertAlmostEqual(exp, calc, 3, msg=f"cse Check Fails: case={case} exp={exp} != calc={calc}")             for exp, calc in zip(self.cases[case]['expected'][4], ret[4]) ]
#             [ self.assertAlmostEqual(exp, calc, 3, msg=f"ScTP Check Fails: case={case} exp={exp} != calc={calc}")            for exp, calc in zip(self.cases[case]['expected'][5], ret[5]) ]
#             [ self.assertAlmostEqual(exp, calc, 3, msg=f"ScFP Check Fails: case={case} exp={exp} != calc={calc}")            for exp, calc in zip(self.cases[case]['expected'][6], ret[6]) ]

#             ax[case].set(xlim=(0, 100), xticks=np.arange(0, 100, 5),
#                          ylim=(0, 3), yticks=np.arange(0, 3, 1))
#             ax[case].set_xlabel('Time')
#             ax[case].plot([self.cases[case]['inputs']['ref']['ref_start'][0], self.cases[case]['inputs']['ref']['ref_end'][0]], [2,2], color='red',  marker=".", linestyle='solid', linewidth=1.0, label='Ref')
#             for h in range(len(self.cases[case]['inputs']['tgts'])):
#                 p = 1 - (h*.40)
#                 ax[case].plot([self.cases[case]['inputs']['tgts'][h][0],          self.cases[case]['inputs']['tgts'][h][1] ],       [p,p], color='blue', marker=".", linestyle='solid', linewidth=1.0, label='Sys')
#                 ax[case].plot([ ret[3][0], self.cases[case]['inputs']['tgts'][h][0] ],                                              [p-0.25, p-0.255], color='blue', marker='^', linestyle='dashed', linewidth=1.0, label='Shifted Sys')
#                 ax[case].plot([ self.cases[case]['inputs']['tgts'][h][1], ret[4][0] ],                                              [p-0.25, p-0.25], color='blue', marker='v', linestyle='dashed', linewidth=1.0)
#             ax[case].text(20, 0.3, "%TP={:.3f}, %FP={:.3f}".format(ret[5][0], ret[6][0]))
#             ax[case].text(5, 2.5, "File: {}".format(self.cases[case]['inputs']['file']))
#             ax[case].legend(loc='upper right')
            
#         fig.savefig(os.path.join("/tmp", "IOUTests_v2.png"))
#         plt.close()
            
class IOUTests_v2(unittest.TestCase):
    #  To test the IoU
    # 

    def setUp(self):
        ##############################  IoU     Inter Union   csb   cse   ScTP      ScFP
        self.cases = [ { 'expected': [ [0.75, 0.75],   [30, 30], [40, 40], [20, 20], [60, 60],  [1.0, 1.0],    [0.333, 0.333] ] ,  'inputs': { 'file': 'F4', 'ref':  [ [ 25, 55 ], [ 25, 55 ] ], 'hyp': [20, 60] } }, 
                       { 'expected': [ [0.75],   [30], [40], [20], [60],  [1.0],    [0.333] ] ,  'inputs': { 'file': 'F4', 'ref':  [ [ 25, 55 ] ], 'hyp': [20, 60] } }, 
                       { 'expected': [ [0.75],   [30], [40], [20], [60],  [1.0],    [0.0] ] ,    'inputs': { 'file': 'F5', 'ref':  [ [ 20, 60 ] ], 'hyp': [25, 55] } }, 
                       { 'expected': [ [0.4286], [30], [70],  [5], [75],  [1.0],    [1.333] ] ,  'inputs': { 'file': 'F6', 'ref':  [ [ 25, 55 ] ], 'hyp': [ 5, 75] } },
                       { 'expected': [ [0.4286], [30], [70], [10], [70],  [0.8571], [0.0] ] ,    'inputs': { 'file': 'F7', 'ref':  [ [  5, 75 ] ], 'hyp': [25, 55] } },
                       { 'expected': [ [0.5454], [30], [55], [20], [75],  [1.0],    [0.5714] ] , 'inputs': { 'file': 'F8', 'ref':  [ [ 20, 55 ] ], 'hyp': [25, 75] } },
                       { 'expected': [ [0.50],   [30], [60], [10], [65],  [0.9],    [0.2] ] ,    'inputs': { 'file': 'F9', 'ref':  [ [  5, 55 ] ], 'hyp': [25, 65] } },
                       { 'expected': [ [0.5454], [30], [55], [20], [70],  [0.9],    [0.1] ] ,    'inputs': { 'file': 'F10', 'ref': [ [ 25, 75 ] ], 'hyp': [20, 55] } },
                       { 'expected': [ [0.1667], [10], [60],  [5], [50],  [0.625],  [0.5] ] ,    'inputs': { 'file': 'F11', 'ref': [ [ 25, 65 ] ], 'hyp': [ 5, 35] } }
                     ]
        self.cases = [ { 'expected': [ [0.75],   [30], [40], [20], [60],  [1.0],    [0.333] ] ,  'inputs': { 'file': 'F4', 'ref':  [ [ 25, 55 ] ], 'hyp': [20, 60] } }, 
                       { 'expected': [ [0.75],   [30], [40], [20], [60],  [1.0],    [0.0] ] ,    'inputs': { 'file': 'F5', 'ref':  [ [ 20, 60 ] ], 'hyp': [25, 55] } },
                       { 'expected': [ [0.4286], [30], [70],  [5], [75],  [1.0],    [1.333] ] ,  'inputs': { 'file': 'F6', 'ref':  [ [ 25, 55 ] ], 'hyp': [ 5, 75] } },
                       { 'expected': [ [0.4286], [30], [70], [10], [70],  [0.8571], [0.0] ] ,    'inputs': { 'file': 'F7', 'ref':  [ [  5, 75 ] ], 'hyp': [25, 55] } },
                       { 'expected': [ [0.5454], [30], [55], [20], [75],  [1.0],    [0.5714] ] , 'inputs': { 'file': 'F8', 'ref':  [ [ 20, 55 ] ], 'hyp': [25, 75] } },
                       { 'expected': [ [0.50],   [30], [60], [10], [65],  [0.9],    [0.2] ] ,    'inputs': { 'file': 'F9', 'ref':  [ [  5, 55 ] ], 'hyp': [25, 65] } },
                       { 'expected': [ [0.5454], [30], [55], [20], [70],  [0.9],    [0.1] ] ,    'inputs': { 'file': 'F10', 'ref': [ [ 25, 75 ] ], 'hyp': [20, 55] } },
                       { 'expected': [ [0.1667], [10], [60],  [5], [50],  [0.625],  [0.5] ] ,    'inputs': { 'file': 'F11', 'ref': [ [ 25, 65 ] ], 'hyp': [ 5, 35] } },
                       { 'expected': [ [0.75, 0.4],   [30, 20], [40, 50], [20, 20], [60, 70],  [1.0, 1.0],    [0.333, 0.6667] ] ,  'inputs': { 'file': 'Multi', 'ref':  [ [ 25, 55 ], [ 40, 70 ] ], 'hyp': [20, 60] } },
                       ]
                                                
    def test_iou(self):
        fig, ax = plt.subplots(len(self.cases), 1, figsize=(8,1.5 * len(self.cases)), constrained_layout=True)
        if (len(self.cases) == 1):
            ax = [ax]
        for case in range(len(self.cases)):
            #print(f"==== Begin Case {case} file {self.cases[case]['inputs']['file']}")

            d = pd.DataFrame(self.cases[case]['inputs']['ref'], columns = ['start', 'end'])

            ret = segment_iou_v2(self.cases[case]['inputs']['hyp'][0],
                                 self.cases[case]['inputs']['hyp'][1],
                                 [ d.start, d.end],
                                 15)
            #print("IoU  [0]: " + str(ret[0].to_list()))
            #print("Int  [1]: " + str(ret[1].to_list()))
            #print("Uni  [2]: " + str(ret[2].to_list()) + " type " + str(type(ret[2])) )
            #print("csb  [3]: " + str(ret[3]))
            #print("cse  [4]: " + str(ret[4]))
            #print("ScTP [5]: " + str(ret[5]))
            #print("ScFP [6]: " + str(ret[6]))
             
            [ self.assertAlmostEqual(exp, calc, 3, msg=f"IoU Check Fails: case={case} exp={exp} != calc={calc}")             for exp, calc in zip(self.cases[case]['expected'][0], ret[0]) ]
            [ self.assertAlmostEqual(exp, calc, 3, msg=f"intersection Check Fails: case={case} exp={exp} != calc={calc}")    for exp, calc in zip(self.cases[case]['expected'][1], ret[1]) ]
            [ self.assertAlmostEqual(exp, calc, 3, msg=f"union Check Fails: case={case} exp={exp} != calc={calc}")           for exp, calc in zip(self.cases[case]['expected'][2], ret[2]) ]
            [ self.assertAlmostEqual(exp, calc, 3, msg=f"csb Check Fails: case={case} exp={exp} != calc={calc}")             for exp, calc in zip(self.cases[case]['expected'][3], ret[3]) ]
            [ self.assertAlmostEqual(exp, calc, 3, msg=f"cse Check Fails: case={case} exp={exp} != calc={calc}")             for exp, calc in zip(self.cases[case]['expected'][4], ret[4]) ]
            [ self.assertAlmostEqual(exp, calc, 3, msg=f"ScTP Check Fails: case={case} exp={exp} != calc={calc}")            for exp, calc in zip(self.cases[case]['expected'][5], ret[5]) ]
            [ self.assertAlmostEqual(exp, calc, 3, msg=f"ScFP Check Fails: case={case} exp={exp} != calc={calc}")            for exp, calc in zip(self.cases[case]['expected'][6], ret[6]) ]

            ax[case].set(xlim=(0, 150), xticks=np.arange(0, 100, 5),
                         ylim=(0, 3), yticks=np.arange(0, 3, 1))
            ax[case].set_xlabel('Time')
            is_multi = (len(self.cases[case]['inputs']['ref']) > 1)
            for r in range(len(self.cases[case]['inputs']['ref'])):
                p = 2 + (r*.40)
                ax[case].plot([self.cases[case]['inputs']['ref'][r][0], self.cases[case]['inputs']['ref'][r][1]], [p,p], color='red',  marker=".", linestyle='solid', linewidth=1.0, label=f'Ref {r}' if (is_multi) else 'Ref')
            ax[case].plot([self.cases[case]['inputs']['hyp'][0],        self.cases[case]['inputs']['hyp'][1] ],   [1.5,1.5], color='blue', marker=".", linestyle='solid', linewidth=1.0, label='Sys') 
            for h in range(len(self.cases[case]['expected'][0])):
                p = 1 - (h*.40)
                ax[case].plot([ ret[3][h], self.cases[case]['inputs']['hyp'][0] ],                                 [p-0.0, p-0.0], color='blue', marker='^', linestyle='dashed', linewidth=1.0, label=f'Shifted Sys(Ref#{h})' if (is_multi) else 'Shifted Sys')
                ax[case].plot([ self.cases[case]['inputs']['hyp'][1], ret[4][h] ],                                 [p-0.0, p-0.0], color='blue', marker='^', linestyle='dashed', linewidth=1.0)

                ax[case].text(ret[4][h]+5, p-0.255, "%TP={:.3f}, %FP={:.3f}".format(ret[5][0], ret[6][0]))
            ax[case].text(5, 2.5, "File: {}".format(self.cases[case]['inputs']['file']))
            ax[case].legend(loc='upper right')
            
        fig.savefig(os.path.join("/tmp", "IOUTests_v2.png"))
        plt.close()
            
            
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
            generated_list = merge_vote_time_periods(input_dict, "emotion")
                
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
            input_df.start = input_df.start.astype(str)
            input_df.end = input_df.end.astype(str)
            input_df.Class = input_df.Class.astype(str)
      
            # Call get_highest_vote_based_on_time w/ input data frame
            is_emotion = False
            id_counter = 0
            last_segment_id = ""
            for index, row in input_df.iterrows():
                current_segment_id = row['segment_id']
                if current_segment_id == last_segment_id:
                    continue
                else:
                    partial_df = input_df[input_df['segment_id'] == current_segment_id]
                    user_ids = partial_df.user_id.values.tolist()
                    if len(set(user_ids)) > 1:
                        is_emotion = True
                        break
                    else:
                        last_segment_id = current_segment_id

            if is_emotion == False:
                emo_dict = get_highest_vote_based_on_time(input_df, "norm")
            else:
                emo_dict = get_highest_vote_based_on_time(input_df, "emotion")

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
