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
    print("gen:{} exp:{}".format(generated, expected))
    ### If update is set and the file is missing, add the generated file    
    if (not os.path.exists(expected)):
        if (os.environ.get("CCUTEST_UPDATE_SCORES") == "add_missing_file"):
            if (not os.path.exists(os.path.dirname(expected))):
                subprocess.check_call("mkdir -p {}".format(os.path.dirname(expected)), shell=True)
            subprocess.check_call(f"cp {generated} {expected}", shell=True)

    if (not os.path.exists(generated)):
        print(f"Error: file {generated} does not exist")
    if (not os.path.exists(expected)):
        print(f"Error: file {expected} does not exist")

    tst = filecmp.cmp(generated, expected)
    if (not tst and (os.environ.get("CCUTEST_UPDATE_SCORES") == "update_failed_file")):
        # The use said to update the file, go ahead.  git will show the difference
        subprocess.check_call(f"cp {generated} {expected}", shell=True)
        #os.remove(generated)
    else:
        if not tst:
            print(f"Files differ.  Use the command:\ntkdiff {generated}\\\n   {expected}")
            subprocess.check_call(f"sdiff -w230 {generated} {expected}", shell=True)
            assert tst, "Files differ"
        #else:
        #    os.remove(generated)
            

def run_scorer(tmp_dir):
    print("Scoring Command: python -m CCU_validation_scoring.cli " + " ".join(sys.argv[1:]))
    try:
        rtn = cli.main()
        print(f"Scorer-produced files in {tmp_dir}\n   {os.listdir(tmp_dir)}")

    except Exception:
        print("Scorer execution threw and exception")
        exit(1)

    
@pytest.mark.parametrize("dataset, system_input_index, system_dir, task, opt1, opt2, opt3, score_tag",
                         [('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.VD.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'VD', '', '', '', ''),
                          ('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.VD.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'VD_gaps', '', '', '-g', ''),
                          ('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.AD.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'AD', '', '', '', ''),
                          ('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'ND', '', '', '', ''),
                          ('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.ED.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'ED', '', '', '', ''),
                          ('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.CD.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'CD', '', '', '', ''),
                          ('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'ND_open', '', '', '-sf open', ''),
                          ('LDC_reference_sample', 'LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab', 'pass_submissions_LDC_reference_sample', 'NDMAP', 'CCU_P1_TA1_ND_NIST_mini-eval1_20220531_050236', '', '', ''),
                          
                          ('LC1-SimulatedMiniEvalP1_ref_annotation', 'LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation', 'ND', '', 'known_norms_LC1.txt', '', ''),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation', 'LC1-SimulatedMiniEvalP1.20220909.CD.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation', 'CD', '', '', '', ''),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation', 'LC1-SimulatedMiniEvalP1.20220909.ED.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation', 'ED', '', '', '', ''),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation', 'LC1-SimulatedMiniEvalP1.20220909.VD.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation', 'VD', '', '', '', ''),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation', 'LC1-SimulatedMiniEvalP1.20220909.AD.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation', 'AD', '', '', '', ''),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation', 'LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation', 'NDMAP', 'CCU_P1_TA1_ND_NIST_mini-eval1_20220908_111111', 'hidden_norms_LC1.txt', '', ''),

                          ('LC1-SimulatedMiniEvalP1_ref_annotation', 'LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation', 'ND', '', 'known_norms_LC1.txt', '-lf after_read:by_value:0.74',       'llrfilt_after_read'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation', 'LC1-SimulatedMiniEvalP1.20220909.CD.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation', 'CD', '', '',                    '-lf after_read:by_value:0.74',       'llrfilt_after_read'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation', 'LC1-SimulatedMiniEvalP1.20220909.ED.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation', 'ED', '', '',                    '-lf after_read:by_value:0.74',       'llrfilt_after_read'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation', 'LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation', 'ND', '', 'known_norms_LC1.txt', '-lf after_transforms:by_value:0.74', 'llrfilt_after_transforms'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation', 'LC1-SimulatedMiniEvalP1.20220909.CD.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation', 'CD', '', '',                    '-lf after_transforms:by_value:0.74', 'llrfilt_after_transforms'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation', 'LC1-SimulatedMiniEvalP1.20220909.ED.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation', 'ED', '', '',                    '-lf after_transforms:by_value:0.74', 'llrfilt_after_transforms'),

                          ('ActEV-SmoothCurve', 'ActEV-SmoothCurve.ND.scoring.index.tab', 'pass_submissions_ActEV-SmoothCurve', 'ND', '', '', '', ''),

                          ('LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'ND', '', 'known_norms_LC1.txt', '',                                                                   'nomerge'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'ND', '', 'known_norms_LC1.txt', '-aS 3 -lS min_llr -vS class-status -xS 30',                          'merge-min_llr-class-status'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'ND', '', 'known_norms_LC1.txt', '-aS 3 -lS min_llr -vS class -xS 30',                                 'merge-min_llr-class'),

                          ('LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'LC1-SimulatedMiniEvalP1.20220909.ED.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'ED', '', '',                    '',                                                                   'nomerge'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'LC1-SimulatedMiniEvalP1.20220909.ED.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'ED', '', '',                    '-aR 0 -xR 0 -mv 1',                                                  'vote1'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'LC1-SimulatedMiniEvalP1.20220909.ED.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'ED', '', '',                    '-aR 0 -xR 0 -mv 2',                                                  'vote2'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'LC1-SimulatedMiniEvalP1.20220909.ED.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'ED', '', '',                    '-aR 0 -xR 0 -mv 3',                                                  'vote3'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'LC1-SimulatedMiniEvalP1.20220909.ED.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'ED', '', '',                    '-aS 3 -lS min_llr -vS class -xS 30',                                 'merge-min_llr-class'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'LC1-SimulatedMiniEvalP1.20220909.ED.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'ED_filemerge', '', '',                    '-d -fm 1', 'fullfilemerge'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'LC1-SimulatedMiniEvalP1.20220909.ED.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'ED_filemerge', '', '',                    '-d -fm 1/2', 'halffilemerge'),
                          ('LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'LC1-SimulatedMiniEvalP1.20220909.ED.scoring.index.tab', 'pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation_merge', 'ED_filemerge', '', '',                    '-d -fm 1/4', 'quarterfilemerge'),
                          
                          ### These use class 105 to explore the merging variations
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests.txt', '', 'AlignFile-105-noopt'),
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests.txt', '-aR 30 -xR 300 -vR class', 'AlignFile-105-refM_class'),
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests.txt', '-aR 30 -xR 300 -vR class-status', 'AlignFile-105-refM_class_status'),
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests.txt', '-aS 0.4 -xS 30 -vS class -lS max_llr ', 'AlignFile-105-sysM_class'),
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests.txt', '-aS 0.4 -xS 30 -vS class-status -lS max_llr', 'AlignFile-105-sysM_class_status'),
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests.txt', '-aR 30 -xR 300 -vR class-status -aS 0.4 -xS 30 -vS class-status -lS max_llr', 'AlignFile-105-refM_class_status_sysM_class_status'),

                          ### These use class 102 to test the splits for the ManyRef:OneHyp experiments
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests_102.txt', '-aC 15 -xC 150 -t intersection:gt:0', 'AlignFile-102-noopt-noRefM-noSysM'),
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests_102.txt', '-aC 15 -xC 150 -t intersection:gt:0 --align_hacks ManyRef:OneHyp', 'AlignFile-102-MROH-noRefM-noSysM'),
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests_102.txt', '-aC 15 -xC 150 -t intersection:gt:0 --align_hacks OneRef:ManyHyp', 'AlignFile-102-ORMH-noRefM-noSysM'),
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests_102.txt', '-aC 15 -xC 150 -t intersection:gt:0 --align_hacks ManyRef:ManyHyp', 'AlignFile-102-MRMH-noRefM-noSysM'),
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests_102.txt', '-aC 15 -xC 150 -t intersection:gt:0 --align_hacks ManyRef:OneHyp -aR 30 -xR 300 -vR class', 'AlignFile-102-MROH-RefM-noSysM'),
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests_102.txt', '-aC 15 -xC 150 -t intersection:gt:0 --align_hacks ManyRef:OneHyp -aS 30 -xS 300 -vR class -lS max_llr -vS class -aR 30 -xR 300', 'AlignFile-102-MROH-RefM-SysM'),

                          ### These use class 108 to test noscore handling changes
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests_108.txt', '-aC 15 -xC 150 -t intersection:gt:0', 'AlignFile-108-noopt'),
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'ND', '', 'known_norms_AlignFile_tests_108.txt', '-aC 15 -xC 150 -t intersection:gt:0 -aS 30 -xS 300 -lS max_llr -vS class -aR 300 -xR 3000 -vR class', 'AlignFile-108-SysMerge'),

                          ## Test for CD
                          ('AlignFile_tests', 'AlignFile_tests.scoring_input.index.tab', 'pass_submissions_AlignFile_tests', 'CD', '', 'known_norms_AlignFile_tests.txt', '', 'AlignFile-CD-noopt'),

                          ('WeightedF1', 'WeightedF1.scoring_input.index.tab', 'pass_submissions_WeightedF1', 'ND', '', 'known_norms_WeightedF1.txt', '', ''),
                          ('WeightedF1', 'WeightedF1.scoring_input.index.tab', 'pass_submissions_WeightedF1', 'ND', '', 'known_norms_WeightedF1.txt', '-t intersection:gt:0', 'any_overlap'),
                          ('WeightedF1', 'WeightedF1.scoring_input.index.tab', 'pass_submissions_WeightedF1', 'ND', '', 'known_norms_WeightedF1.txt', '-aR 30 -xR 300 -vR class -aC 15 -xC 150 -t intersection:gt:0', 'LC1_Eval_V1_RefMerge_NoSysMerge')
                          ])

def test_run_score_submissions(dataset, system_input_index, system_dir, task, opt1, opt2, opt3, score_tag):
    def clean_tmp_dir(tmp_dir):
        for myfile in os.listdir(tmp_dir):
            if (myfile not in ['.gitignore']):
                try:
                    os.remove(os.path.join(tmp_dir, myfile))
                except OSError as e:
                    # If it fails, inform the user.
                    print(f"Error: Could not remove {tmp_dir}/{myfile}")
                    exit(1)

    current_path = os.path.abspath(__file__)
    test_dir_path = os.path.dirname(current_path)
    
    submissions_path = os.path.join(test_dir_path, 'pass_submissions')
    reference_path = os.path.join(test_dir_path, 'reference')
    score_path = os.path.join(test_dir_path, 'scores')
    tmp_dir = os.path.join(test_dir_path, 'test_output')

    refdir = os.path.join(reference_path, dataset)
    scoring_index_path = os.path.join(refdir, 'index_files', system_input_index)
    subdirs = glob.glob(os.path.join(submissions_path, system_dir, task, '*'))

    print("/n")
    for s in subdirs:
        print(f"SUBDIR: {s}")

    if (task in ['AD', 'VD', 'VD_gaps']):
        assert len(subdirs) > 0        
        for subdir in subdirs:
            if task == "VD_gaps":
                sys.argv[1:] = ["score-{}".format("vd"), "-ref", refdir,
                            "-s", subdir, "-i", scoring_index_path, "-o", tmp_dir]
            else:                       
                sys.argv[1:] = ["score-{}".format(task.lower()), "-ref", refdir,
                            "-s", subdir, "-i", scoring_index_path, "-o", tmp_dir]
            clean_tmp_dir(tmp_dir)

            if (opt3 != ''):
                sys.argv.extend(opt3.split())
            run_scorer(tmp_dir)

#            for filename in ["scores_aggregated.tab", "segment_diarization.tab"]:
            for filename in ["segment_diarization.tab"]:                
                tmp_file = os.path.join(tmp_dir, filename)
                tmp_out_dir_name = os.path.basename(subdir) + ('' if (score_tag == '') else '-' + score_tag)
                byte_compare_file(tmp_file,
                                  os.path.join(score_path, "scores_" + dataset, task, tmp_out_dir_name, filename))
            clean_tmp_dir(tmp_dir)


    if (task in ["ND", "ED", "ED_filemerge", "ND_open"]):
        assert len(subdirs) > 0
        
        for subdir in subdirs:
            if task == "ED":            
                sys.argv[1:] = ["score-{}".format(task.lower()), "-ref", refdir,
                                "-s", subdir, "-i", scoring_index_path, "-o", tmp_dir, "-aR", "1.0", "-xR", "10"]
            if task == "ED_filemerge":
                sys.argv[1:] = ["score-{}".format("ed"), "-ref", refdir,
                                "-s", subdir, "-i", scoring_index_path, "-o", tmp_dir]
            if task == "ND":
                sys.argv[1:] = ["score-{}".format(task.lower()), "-ref", refdir,
                                "-s", subdir, "-i", scoring_index_path, "-o", tmp_dir, "-aR", "1.0", "-xR", "10", "-vR", "class"]

            if task == "ND_open":
                sys.argv[1:] = ["score-{}".format("nd"), "-ref", refdir,
                                "-s", subdir, "-i", scoring_index_path, "-o", tmp_dir]
            clean_tmp_dir(tmp_dir)

            ### Add opt2 if there
            if (opt2 != ''):
                sys.argv.append("-n")
                sys.argv.append(os.path.join(test_dir_path, opt2))

            if (opt3 != ''):
                sys.argv.extend(opt3.split())
            run_scorer(tmp_dir)
                
            for filename in ["instance_alignment.tab", "scores_by_class.tab", "scores_aggregated.tab"]:
                tmp_file = os.path.join(tmp_dir, filename)
                tmp_out_dir_name = os.path.basename(subdir) + ('' if (score_tag == '') else '-' + score_tag)
                byte_compare_file(tmp_file,
                                  os.path.join(score_path, "scores_" + dataset, task, tmp_out_dir_name, filename))
            clean_tmp_dir(tmp_dir)

    if (task in ["NDMAP"]):
        assert len(subdirs) > 0
        
        for subdir in subdirs:
            orig_system = glob.glob(os.path.join(submissions_path + "/" + system_dir, "ND", opt1))
            print(f"Original System from {submissions_path}/{system_dir}/ND/{opt1}:" + str(orig_system))
            assert (len(orig_system) == 1), "Error:  NDMAP system is not uniq"
                    
            sys.argv[1:] = ["score-nd", "-ref", refdir, "-s", orig_system[0],
                            "-m", subdir, "-i", scoring_index_path, "-o", tmp_dir, "-aR", "1.0", "-xR", "10", "-vR", "class"]
            ### Add opt2 if there
            if (opt2 != ''):
                sys.argv.append("-n")
                sys.argv.append(os.path.join(test_dir_path, opt2))
            clean_tmp_dir(tmp_dir)
            run_scorer(tmp_dir)
            
            for filename in [ "instance_alignment.tab", "scores_by_class.tab", "scores_aggregated.tab"]:
                tmp_file = os.path.join(tmp_dir, filename)
                tmp_out_dir_name = os.path.basename(subdir) + ('' if (score_tag == '') else '-' + score_tag)
                byte_compare_file(tmp_file,
                                  os.path.join(score_path, "scores_" + dataset, task, tmp_out_dir_name, filename))
            clean_tmp_dir(tmp_dir)
                
    if (task in ["CD"]):
        assert len(subdirs) > 0
        for subdir in subdirs:
            sys.argv[1:] = ["score-{}".format(task.lower()), "-ref", refdir,
                            "-s", subdir, "-i", scoring_index_path, "-o", tmp_dir]

            clean_tmp_dir(tmp_dir)
            run_scorer(tmp_dir)
                
            for filename in ["instance_alignment.tab", "scores_by_class.tab"]:
                tmp_file = os.path.join(tmp_dir, filename)
                tmp_out_dir_name = os.path.basename(subdir) + ('' if (score_tag == '') else '-' + score_tag)
                byte_compare_file(tmp_file,
                                  os.path.join(score_path, "scores_" + dataset, task, tmp_out_dir_name, filename))
            clean_tmp_dir(tmp_dir)


if __name__ == '__main__':
    unittest.main()

