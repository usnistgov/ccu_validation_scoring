# import fad21
# from fad21.scoring import score_ac
# from fad21.validation import validate_ac
# from fad21.datatypes import Dataset
# from fad21.io import *
#import CCU_validation_scoring
import CCU_validation_scoring
from CCU_validation_scoring.score_submission import score_nd_submission_dir_cli

from pathlib import Path
import pytest
import io
import os
import pandas as pd

# Validates scorer-implementation w/ specific classification use-cases.

def get_root():
    return Path(__file__).parent

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def scoring_run(refFile, hypFile, iou_threshold, out_dir):
    """ Loader Func """
    
    args = Namespace(reference_dir=refFile, submission_dir=hypFile, mapping_submission_dir=None, iou_thresholds=iou_threshold, output_dir=out_dir)   
    score_nd_submission_dir_cli(args);
    class_df = pd.read_csv(os.path.join(args.output_dir, 'class_scores.csv'))
    agg_df = pd.read_csv(os.path.join(args.output_dir, 'system_scores.csv'))

    return([class_df, agg_df])

 
def test_nd_first(tmpdir):
    """ USE-CASES
    """

    class_df, agg_df = scoring_run('test/reference/LDC_reference_sample', 'test/submission/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220815_164236', "0.01", tmpdir)
    
    print(agg_df)
    print(class_df)
    assert(agg_df.iloc[0,2] == pytest.approx(1.0, 0.1))
    assert(class_df.iloc[1,4] == pytest.approx(1.0, 0.1))
