# -*- coding: utf-8 -*-
'''
    This module implements unit tests of the submission scoring.
'''

import os, sys, glob, filecmp
import pandas as pd
import unittest
import pytest
import subprocess

from CCU_validation_scoring import cli


def byte_compare_file(generated, expected):
    ### If update is set and the file is missing, add the generated file
    if (not os.path.exists(expected)):
        if (os.environ.get("CCUTEST_UPDATE_SCORES") == "add_missing_file"):
            subprocess.check_call(f"cp {generated} {expected}", shell=True)

    if (not os.path.exists(generated)):
        print(f"Error: file {generated} does not exist")
    if (not os.path.exists(expected)):
        print(f"Error: file {expected} does not exist")

    tst = filecmp.cmp(generated, expected)
    if (not tst and (os.environ.get("CCUTEST_UPDATE_SCORES") == "update_failed_file")):
        # The use said to update the file, go ahead.  git will show the difference
        subprocess.check_call(f"cp {generated} {expected}", shell=True)
    else:
        if not tst:
            print(f"Files differ.  Use the command:\ndiff {generated}\\\n   {expected}")
            assert tst, "Files differ"

def run_scorer():
    try:
        cli.main()
        
    except Exception:
        print("Scorer test failed with command")

    
@pytest.mark.parametrize("dataset, system_input_index, system_dir, task, opt1",
                         [('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'VD', ''),
                          ('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'AD', ''),
                          ('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'ND', ''),
                          ('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'ED', ''),
                          ('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'CD', ''),
                          ('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'NDMAP', ''),

                          ('LC1-SimulatedMiniEvalP1_ref_annotation', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation', 'ND', '')                        

                          ])
def test_run_score_submissions(dataset, system_input_index, system_dir, task, opt1):
    current_path = os.path.abspath(__file__)
    test_dir_path = os.path.dirname(current_path)
    
    submissions_path = os.path.join(test_dir_path, 'pass_submissions')
    reference_path = os.path.join(test_dir_path, 'reference')
    score_path = os.path.join(test_dir_path, 'scores')
    tmp_dir = test_dir_path

    refdir = os.path.join(reference_path, dataset)
    scoring_index_path = os.path.join(refdir, 'index_files', system_input_index)
    subdirs = glob.glob(os.path.join(submissions_path, system_dir, task, '*'))

    print("/n")
    for s in subdirs:
        print(f"SUBDIR: {s}")
    
    if (task in ['AD', 'VD']):
        assert len(subdirs) > 0        
        for subdir in subdirs:                        
            sys.argv[1:] = ["score-{}".format(task.lower()), "-ref", refdir,
                            "-s", subdir, "-i", scoring_index_path, "-o", tmp_dir]
            run_scorer()

            for filename in ["scores_aggregated.tab", "segment_diarization.tab"]:
                tmp_file = os.path.join(tmp_dir, filename)
                byte_compare_file(tmp_file,
                                  os.path.join(score_path, "scores_" + dataset, task, os.path.basename(subdir), filename))
                os.remove(tmp_file)


    if (task in ["ND", "ED"]):
        assert len(subdirs) > 0
        
        for subdir in subdirs:            
            sys.argv[1:] = ["score-{}".format(task.lower()), "-ref", refdir,
                            "-s", subdir, "-i", scoring_index_path, "-o", tmp_dir]
            run_scorer()
                
            for filename in ["scores_by_class.tab", "scores_aggregated.tab", "instance_alignment.tab"]:
                tmp_file = os.path.join(tmp_dir, filename)
                byte_compare_file(tmp_file,
                                  os.path.join(score_path, "scores_" + dataset, task, os.path.basename(subdir), filename))
                os.remove(tmp_file)

    if (task in ["NDMAP"]):
        assert len(subdirs) > 0
        
        for subdir in subdirs:
            sys.argv[1:] = ["score-nd", "-ref", refdir, "-s", subdir,
                            "-m", subdir, "-i", scoring_index_path, "-o", tmp_dir]
            run_scorer()
            
            for filename in ["scores_by_class.tab", "scores_aggregated.tab", "instance_alignment.tab"]:
                tmp_file = os.path.join(tmp_dir, filename)
                byte_compare_file(tmp_file,
                                  os.path.join(score_path, "scores_" + dataset, task, os.path.basename(subdir), filename))
                #os.remove(tmp_file)
                
    if (task in ["CD"]):
        assert len(subdirs) > 0
        for subdir in subdirs:
            sys.argv[1:] = ["score-{}".format(task.lower()), "-ref", refdir,
                            "-s", subdir, "-i", scoring_index_path, "-o", tmp_dir]

            run_scorer()
                
            for filename in ["scores_by_class.tab", "instance_alignment.tab"]:
                tmp_file = os.path.join(tmp_dir, filename)
                byte_compare_file(tmp_file,
                                  os.path.join(score_path, "scores_" + dataset, task, os.path.basename(subdir), filename))
                os.remove(tmp_file)


        # dirs = ["CD"]
        # for dir in dirs:

        #     refdir = os.path.join(self.reference_path, 'LC1-SimulatedMiniEvalP1_ref_annotation')
        #     self.scoring_index_path = os.path.join(refdir, 'index_files', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab')
        #     subdirs = glob.glob(os.path.join(self.submissions_path, "pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation", dir, '*'))
            
        #     assert len(subdirs) > 0

        #     for subdir in subdirs:

        #         tmp_alignment_file = os.path.join(self.tmp_dir, "instance_alignment.tab")

        #         score_ref_alignment_file = os.path.join(self.score_path, "scores_LC1-SimulatedMiniEvalP1_ref_annotation", dir, os.path.basename(subdir), "instance_alignment.tab")
        #         score_ref_alignment_df = pd.read_csv(score_ref_alignment_file, dtype={"class": object}, sep = "\t")
        #         score_ref_alignment_df_sorted = score_ref_alignment_df.sort_values(by=['class', 'file_id', 'sys', 'ref'])
        #         score_ref_alignment_df_sorted.to_csv(os.path.join(self.tmp_dir, "instance_alignment_ref.tab"), index = False, quoting=3, sep="\t", escapechar="\t")

        #         sys.argv[1:] = ["score-{}".format(dir.lower()), "-ref", refdir,
        #                     "-s", subdir, "-i", self.scoring_index_path, "-o", self.tmp_dir]
        #         try:
        #             cli.main()
        #             self.assertTrue(filecmp.cmp(os.path.join(self.tmp_dir, "instance_alignment_ref.tab"), tmp_alignment_file))
        #             os.remove(tmp_alignment_file)
        #             os.remove(os.path.join(self.tmp_dir, "scores_by_class.tab"))
        #             os.remove(os.path.join(self.tmp_dir, "instance_alignment_ref.tab"))

        #         except Exception:
        #             self.fail("Scorer test failed on submission {} with reference {}".format(subdir, self.reference_path))

if __name__ == '__main__':
    unittest.main()

