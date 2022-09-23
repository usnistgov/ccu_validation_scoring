# -*- coding: utf-8 -*-
'''
    This module implements unit tests of the submission scoring.
'''

import os, sys, glob, filecmp
import pandas as pd
import unittest


from CCU_validation_scoring import cli


class TestScoreSubmission(unittest.TestCase):
    """
        Test submissions
    """

    def setUp(self):
        current_path = os.path.abspath(__file__)
        test_dir_path = os.path.dirname(current_path)
        
        self.submissions_path = os.path.join(test_dir_path, 'pass_submissions')
        self.reference_path = os.path.join(test_dir_path, 'reference')
        self.score_path = os.path.join(test_dir_path, 'scores')
        self.tmp_dir = test_dir_path

    def test_score_submissions(self):

        dirs = ["AD", "VD"]
        for dir in dirs:

            refdir = os.path.join(self.reference_path, 'LDC_reference_sample')
            self.scoring_index_path = os.path.join(refdir, 'index_files', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab')
            subdirs = glob.glob(os.path.join(self.submissions_path, "pass_submissions_LDC_reference_sample", dir, '*'))
            
            assert len(subdirs) > 0

            for subdir in subdirs:

                tmp_system_file = os.path.join(self.tmp_dir, "scores_aggregated.tab")
                tmp_segment_file = os.path.join(self.tmp_dir, "segment_diarization.tab")

                score_ref_system_file = os.path.join(self.score_path, "scores_LDC_reference_sample", dir, os.path.basename(subdir), "scores_aggregated.tab")
                score_ref_segment_file = os.path.join(self.score_path, "scores_LDC_reference_sample", dir, os.path.basename(subdir), "segment_diarization.tab")

                sys.argv[1:] = ["score-{}".format(dir.lower()), "-ref", refdir,
                            "-s", subdir, "-i", self.scoring_index_path, "-o", self.tmp_dir]
                try:
                    cli.main()
                    self.assertTrue(filecmp.cmp(score_ref_system_file, tmp_system_file))
                    os.remove(tmp_system_file)
                    self.assertTrue(filecmp.cmp(score_ref_segment_file, tmp_segment_file))
                    os.remove(tmp_segment_file)

                except Exception:
                    self.fail("Scorer test failed on submission {} with reference {}".format(subdir, self.reference_path))

        dirs = ["ND", "ED"]
        for dir in dirs:

            refdir = os.path.join(self.reference_path, 'LDC_reference_sample')
            self.scoring_index_path = os.path.join(refdir, 'index_files', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab')
            subdirs = glob.glob(os.path.join(self.submissions_path, "pass_submissions_LDC_reference_sample", dir, '*'))
            
            assert len(subdirs) > 0

            for subdir in subdirs:

                tmp_class_file = os.path.join(self.tmp_dir, "scores_by_class.tab")
                tmp_system_file = os.path.join(self.tmp_dir, "scores_aggregated.tab")
                tmp_alignment_file = os.path.join(self.tmp_dir, "instance_alignment.tab")

                score_ref_class_file = os.path.join(self.score_path, "scores_LDC_reference_sample", dir, os.path.basename(subdir), "scores_by_class.tab")
                score_ref_system_file = os.path.join(self.score_path, "scores_LDC_reference_sample", dir, os.path.basename(subdir), "scores_aggregated.tab")
                score_ref_alignment_file = os.path.join(self.score_path, "scores_LDC_reference_sample", dir, os.path.basename(subdir), "instance_alignment.tab")

                sys.argv[1:] = ["score-{}".format(dir.lower()), "-ref", refdir,
                            "-s", subdir, "-i", self.scoring_index_path, "-o", self.tmp_dir]
                try:
                    cli.main()
                    self.assertTrue(filecmp.cmp(score_ref_class_file, tmp_class_file))
                    os.remove(tmp_class_file)
                    self.assertTrue(filecmp.cmp(score_ref_system_file, tmp_system_file))
                    os.remove(tmp_system_file)
                    self.assertTrue(filecmp.cmp(score_ref_alignment_file, tmp_alignment_file))
                    os.remove(tmp_alignment_file)

                except Exception:
                    self.fail("Scorer test failed on submission {} with reference {}".format(subdir, self.reference_path))

        dirs = ["CD"]
        for dir in dirs:

            refdir = os.path.join(self.reference_path, 'LDC_reference_sample')
            self.scoring_index_path = os.path.join(refdir, 'index_files', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab')
            subdirs = glob.glob(os.path.join(self.submissions_path, "pass_submissions_LDC_reference_sample", dir, '*'))
            
            assert len(subdirs) > 0

            for subdir in subdirs:

                tmp_class_file = os.path.join(self.tmp_dir, "scores_by_class.tab")
                tmp_alignment_file = os.path.join(self.tmp_dir, "instance_alignment.tab")

                score_ref_class_file = os.path.join(self.score_path, "scores_LDC_reference_sample", dir, os.path.basename(subdir), "scores_by_class.tab")
                score_ref_alignment_file = os.path.join(self.score_path, "scores_LDC_reference_sample", dir, os.path.basename(subdir), "instance_alignment.tab")

                sys.argv[1:] = ["score-{}".format(dir.lower()), "-ref", refdir,
                            "-s", subdir, "-i", self.scoring_index_path, "-o", self.tmp_dir]
                try:
                    cli.main()
                    self.assertTrue(filecmp.cmp(score_ref_class_file, tmp_class_file))
                    os.remove(tmp_class_file)
                    self.assertTrue(filecmp.cmp(score_ref_alignment_file, tmp_alignment_file))
                    os.remove(tmp_alignment_file)

                except Exception:
                    self.fail("Scorer test failed on submission {} with reference {}".format(subdir, self.reference_path))

        dir = "NDMAP"
        nd_submission = 'CCU_P1_TA1_NDMAP_NIST_mini-eval1_20220605_050236'
        refdir = os.path.join(self.reference_path, 'LDC_reference_sample')
        self.scoring_index_path = os.path.join(refdir, 'index_files', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab')
        subdir = os.path.join(self.submissions_path, "pass_submissions_LDC_reference_sample", dir, nd_submission)
            
        tmp_class_file = os.path.join(self.tmp_dir, "scores_by_class.tab")
        tmp_system_file = os.path.join(self.tmp_dir, "scores_aggregated.tab")
        tmp_alignment_file = os.path.join(self.tmp_dir, "instance_alignment.tab")

        score_ref_class_file = os.path.join(self.score_path, "scores_LDC_reference_sample", dir, os.path.basename(subdir), "scores_by_class.tab")
        score_ref_system_file = os.path.join(self.score_path, "scores_LDC_reference_sample", dir, os.path.basename(subdir), "scores_aggregated.tab")
        score_ref_alignment_file = os.path.join(self.score_path, "scores_LDC_reference_sample", dir, os.path.basename(subdir), "instance_alignment.tab")

        sys.argv[1:] = ["score-nd", "-ref", refdir, "-s", "test/pass_submissions/pass_submissions_LDC_reference_sample/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220531_050236",
                    "-m", subdir, "-i", self.scoring_index_path, "-o", self.tmp_dir]
        try:
            cli.main()
            self.assertTrue(filecmp.cmp(score_ref_class_file, tmp_class_file))
            os.remove(tmp_class_file)
            self.assertTrue(filecmp.cmp(score_ref_system_file, tmp_system_file))
            os.remove(tmp_system_file)
            self.assertTrue(filecmp.cmp(score_ref_alignment_file, tmp_alignment_file))
            os.remove(tmp_alignment_file)

        except Exception:
            self.fail("Scorer test failed on submission {} with reference {}".format(subdir, self.reference_path))
        
        dirs = ["ND"]
        for dir in dirs:

            refdir = os.path.join(self.reference_path, 'LC1-SimulatedMiniEvalP1_ref_annotation')
            self.scoring_index_path = os.path.join(refdir, 'index_files', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab')
            subdirs = glob.glob(os.path.join(self.submissions_path, "pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation", dir, '*'))
            
            assert len(subdirs) > 0

            for subdir in subdirs:

                tmp_alignment_file = os.path.join(self.tmp_dir, "instance_alignment.tab")

                score_ref_alignment_file = os.path.join(self.score_path, "scores_LC1-SimulatedMiniEvalP1_ref_annotation", dir, os.path.basename(subdir), "instance_alignment.tab")
                score_ref_alignment_df = pd.read_csv(score_ref_alignment_file, dtype={"class": object}, sep = "\t")
                score_ref_alignment_df_sorted = score_ref_alignment_df.sort_values(by=['class', 'file_id', 'sys', 'ref'])
                score_ref_alignment_df_sorted.to_csv(os.path.join(self.tmp_dir, "instance_alignment_ref.tab"), index = False, quoting=3, sep="\t", escapechar="\t")

                sys.argv[1:] = ["score-{}".format(dir.lower()), "-ref", refdir,
                            "-s", subdir, "-i", self.scoring_index_path, "-o", self.tmp_dir]
                try:
                    cli.main()
                    self.assertTrue(filecmp.cmp(os.path.join(self.tmp_dir, "instance_alignment_ref.tab"), tmp_alignment_file))
                    os.remove(tmp_alignment_file)
                    os.remove(os.path.join(self.tmp_dir, "scores_aggregated.tab"))
                    os.remove(os.path.join(self.tmp_dir, "scores_by_class.tab"))
                    os.remove(os.path.join(self.tmp_dir, "instance_alignment_ref.tab"))

                except Exception:
                    self.fail("Scorer test failed on submission {} with reference {}".format(subdir, self.reference_path))

        dirs = ["CD"]
        for dir in dirs:

            refdir = os.path.join(self.reference_path, 'LC1-SimulatedMiniEvalP1_ref_annotation')
            self.scoring_index_path = os.path.join(refdir, 'index_files', 'LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab')
            subdirs = glob.glob(os.path.join(self.submissions_path, "pass_submissions_LC1-SimulatedMiniEvalP1_ref_annotation", dir, '*'))
            
            assert len(subdirs) > 0

            for subdir in subdirs:

                tmp_alignment_file = os.path.join(self.tmp_dir, "instance_alignment.tab")

                score_ref_alignment_file = os.path.join(self.score_path, "scores_LC1-SimulatedMiniEvalP1_ref_annotation", dir, os.path.basename(subdir), "instance_alignment.tab")
                score_ref_alignment_df = pd.read_csv(score_ref_alignment_file, dtype={"class": object}, sep = "\t")
                score_ref_alignment_df_sorted = score_ref_alignment_df.sort_values(by=['class', 'file_id', 'sys', 'ref'])
                score_ref_alignment_df_sorted.to_csv(os.path.join(self.tmp_dir, "instance_alignment_ref.tab"), index = False, quoting=3, sep="\t", escapechar="\t")

                sys.argv[1:] = ["score-{}".format(dir.lower()), "-ref", refdir,
                            "-s", subdir, "-i", self.scoring_index_path, "-o", self.tmp_dir]
                try:
                    cli.main()
                    self.assertTrue(filecmp.cmp(os.path.join(self.tmp_dir, "instance_alignment_ref.tab"), tmp_alignment_file))
                    os.remove(tmp_alignment_file)
                    os.remove(os.path.join(self.tmp_dir, "scores_by_class.tab"))
                    os.remove(os.path.join(self.tmp_dir, "instance_alignment_ref.tab"))

                except Exception:
                    self.fail("Scorer test failed on submission {} with reference {}".format(subdir, self.reference_path))

if __name__ == '__main__':
    unittest.main()

