# -*- coding: utf-8 -*-
'''
    This module implements unit tests of the submission scoring.
'''

import os, sys, glob, filecmp
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
        self.reference_path = os.path.join(test_dir_path, 'reference', 'LDC_reference_sample')
        self.score_path = os.path.join(test_dir_path, 'scores')
        self.tmp_dir = test_dir_path

    def test_score_submissions(self):

        dirs = ["AD", "VD", "CD"]
        for dir in dirs:
            subdirs = glob.glob(os.path.join(self.submissions_path, dir, '*'))
            
            assert len(subdirs) > 0

            for subdir in subdirs:

                tmp_system_file = os.path.join(self.tmp_dir, "system_scores.csv")

                score_ref_system_file = os.path.join(self.score_path, dir, os.path.basename(subdir), "system_scores.csv")

                sys.argv[1:] = ["score-{}".format(dir.lower()), "-ref", self.reference_path,
                            "-s", subdir, "-o", self.tmp_dir]
                try:
                    cli.main()
                    self.assertTrue(filecmp.cmp(score_ref_system_file, tmp_system_file))
                    os.remove(tmp_system_file)

                except Exception:
                    self.fail("Scorer test failed on submission {} with reference {}".format(subdir, self.reference_path))

        dirs = ["ND", "ED"]
        for dir in dirs:
            subdirs = glob.glob(os.path.join(self.submissions_path, dir, '*'))
            
            assert len(subdirs) > 0

            for subdir in subdirs:

                tmp_class_file = os.path.join(self.tmp_dir, "class_scores.csv")
                tmp_system_file = os.path.join(self.tmp_dir, "system_scores.csv")

                score_ref_class_file = os.path.join(self.score_path, dir, os.path.basename(subdir), "class_scores.csv")
                score_ref_system_file = os.path.join(self.score_path, dir, os.path.basename(subdir), "system_scores.csv")

                sys.argv[1:] = ["score-{}".format(dir.lower()), "-ref", self.reference_path,
                            "-s", subdir, "-o", self.tmp_dir]
                try:
                    cli.main()
                    self.assertTrue(filecmp.cmp(score_ref_class_file, tmp_class_file))
                    self.assertTrue(filecmp.cmp(score_ref_system_file, tmp_system_file))
                    os.remove(tmp_class_file)
                    os.remove(tmp_system_file)

                except Exception:
                    self.fail("Scorer test failed on submission {} with reference {}".format(subdir, self.reference_path))

        dir = "NDMAP"
        nd_submission = 'CCU_P1_TA1_NDMAP_NIST_mini-eval1_20220605_050236'
        subdir = os.path.join(self.submissions_path, dir, nd_submission)
            
        tmp_class_file = os.path.join(self.tmp_dir, "class_scores.csv")
        tmp_system_file = os.path.join(self.tmp_dir, "system_scores.csv")

        score_ref_class_file = os.path.join(self.score_path, dir, nd_submission, "class_scores.csv")
        score_ref_system_file = os.path.join(self.score_path, dir, nd_submission, "system_scores.csv")

        sys.argv[1:] = ["score-nd", "-ref", self.reference_path, "-s", "test/pass_submissions/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220531_050236",
                    "-m", subdir, "-o", self.tmp_dir]
        try:
            cli.main()
            self.assertTrue(filecmp.cmp(score_ref_class_file, tmp_class_file))
            self.assertTrue(filecmp.cmp(score_ref_system_file, tmp_system_file))
            os.remove(tmp_class_file)
            os.remove(tmp_system_file)

        except Exception:
            self.fail("Scorer test failed on submission {} with reference {}".format(subdir, self.reference_path))
        


if __name__ == '__main__':
    unittest.main()

