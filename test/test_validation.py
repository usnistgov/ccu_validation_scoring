# -*- coding: utf-8 -*-
'''
    This module implements unit tests of the submission validation.
'''

import os, sys, glob
import unittest

from CCU_validation_scoring import cli


class TestValidateSubmission(unittest.TestCase):
    """
        Test submissions
    """

    def setUp(self):
        current_path = os.path.abspath(__file__)
        test_dir_path = os.path.dirname(current_path)

        self.pass_submissions_path = os.path.join(test_dir_path, 'pass_submissions')
        self.fail_submissions_path = os.path.join(test_dir_path, 'fail_submissions')
        self.reference_path = os.path.join(test_dir_path, 'reference')
        self.hidden_norm_path = os.path.join(test_dir_path, 'hidden_norms.txt')

    def test_valid_submissions(self):

        dirs = ["AD", "CD", "ED", "ND", "NDMAP", "VD"]
        for dir in dirs:

            refdir = os.path.join(self.reference_path, 'LDC_reference_sample')
            subdirs = glob.glob(os.path.join(self.pass_submissions_path, "pass_submissions_LDC_reference_sample", dir, '*'))

            assert len(subdirs) > 0

            for subdir in subdirs:
                if dir == "NDMAP":
                    sys.argv[1:] = ["validate-{}".format(dir.lower()),
                                    "-s", subdir, "-n", self.hidden_norm_path]
                else:
                    sys.argv[1:] = ["validate-{}".format(dir.lower()), "-ref", refdir,
                                    "-s", subdir]
                try:
                    cli.main()
                except Exception:
                    print(sys.argv[1:])
                    self.fail("Validator failed on valid submission {}".format(subdir))


    def test_invalid_submissions(self):

        dirs = ["AD", "ED", "ND", "NDMAP", "VD"]
        for dir in dirs:

            refdir = os.path.join(self.reference_path, 'LDC_reference_sample')
            subdirs = glob.glob(os.path.join(self.fail_submissions_path, "fail_submissions_LDC_reference_sample", dir, '*'))

            assert len(subdirs) > 0

            for subdir in subdirs:
                if dir == "NDMAP":
                    sys.argv[1:] = ["validate-{}".format(dir.lower()),
                                    "-s", subdir, "-n", self.hidden_norm_path]
                else:
                    sys.argv[1:] = ["validate-{}".format(dir.lower()), "-ref", refdir,
                                    "-s", subdir]
                with self.assertRaises(SystemExit):
                    cli.main()
                    print(sys.argv[1:])
                    self.fail("Validator succedded on invalid submission {}".format(subdir))


if __name__ == '__main__':
    unittest.main()
