# test/README.md

This is the README for the ccu_validation_scoring/test directory.  The
test_scoring.py pytest implementation uses the pytest 'parametrize'
pytest functionality to define individual scoring runs to test and an
envirornment variable 'CCUTEST_UPDATE_SCORES' manages the expected
test case output files of the scorer.

During test case generation, it is difficult to manage the outputs
produced by the scorer.  The CCUTEST_UPDATE_SCORES environment
variable helps manage the stored, expected output.  By default, a run
of pytest runs the test cases.  Depending on the value of
CCUTEST_UPDATE_SCORES, the following occurs:

   % export CCUTEST_UPDATE_SCORES=add_missing_file

     When the test script encounters a missing scoring file, the
     presently produced file is copied into the expected scores
     directory so that the test case will pass.

   % export CCUTEST_UPDATE_SCORES=update_failed_file

     When the test script encounters a scorer output file that fails
     validation, the presently produced file is copied into the
     expected scores directory so that the test case will pass.  This
     supports the use case that a vetted change is complete.

Pytest

- Use this to test without an install of the libraries:
    % python3 -m pytest test

- The option '--capture=no' prints stdout.

