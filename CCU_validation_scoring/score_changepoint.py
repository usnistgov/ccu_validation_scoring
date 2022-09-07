import os
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .utils import *

logger = logging.getLogger('SCORING')


def generate_zero_scores_changepoint(labels, delta_cp_text_thresholds, delta_cp_time_thresholds):

    if len(labels)>0:
        y_text = []
        y_time = []
        for idx, act in enumerate(labels):
            if act == "text":
                y_text.append( [ act, 0.0, [0.0, 0.0], [0.0, 1.0] ])
            else:
                y_time.append( [ act, 0.0, [0.0, 0.0], [0.0, 1.0] ])
        df_text = pd.DataFrame(y_text, columns=['type', 'ap', 'precision', 'recall'])
        df_time = pd.DataFrame(y_time, columns=['type', 'ap', 'precision', 'recall'])
        pr_iou_scores = {}
        for iout in delta_cp_text_thresholds:
            pr_iou_scores[iout] = df_text
        for iout in delta_cp_time_thresholds:
            pr_iou_scores[iout] = df_time

    else:
        y = []
        logger.error("No matching Types found in system output.")
        y.append( [ 'no_macthing_Type', 0.0, [0.0, 0.0], [0.0, 1.0] ])
        df_whole = pd.DataFrame(y, columns=['type', 'ap', 'precision', 'recall'])
        pr_iou_scores = {}
        for iout in delta_cp_text_thresholds + delta_cp_time_thresholds:
            pr_iou_scores[iout] = df_whole
    return pr_iou_scores


def segment_cp(ref_class, tgts):
    """
    Compute a delta distance 

    Parameters
    ----------
    ref_class: int/float
        Timestamp of hyp         
    tgts : 1d array
        Timestamp of ref containing [timestamp X N] times.

    Outputs
    -------
    distance : 1d array
        Delta distance of the N's candidate segments.
    """
    distance = abs(np.subtract(ref_class,tgts[0]))    
    return distance


def compute_cps(row, ref):
    refs = ref.loc[ ref['file_id'] == row.file_id ].copy()    
    # If there are no references for this hypothesis it's IoU is 0/FP
    if len(refs) == 0:
        return pd.DataFrame(data=[[row.type, row.file_id, np.nan, row.Class, row.llr, 0.0]],
            columns=['type', 'file_id', 'Class_ref', 'Class_hyp', 'llr', 'delta_cp'])
    else:        
        refs['delta_cp'] = segment_cp(row.Class, [refs.Class])
        rmin = refs.loc[refs.delta_cp == refs.delta_cp.min()]
        rout = rmin.rename(columns={'Class':'Class_ref'})
        rout[['Class_hyp', 'llr']] = row.Class, row.llr

        return rout

def compute_average_precision_cps(ref, hyp, delta_cp_thresholds):
    """ 
    Compute average precision and precision-recall curve at specific text/time
    delta distance thresholds between ground truth and predictions data frames. If multiple
    predictions occur for the same predicted segment, only the one with highest
    delta distance is matched as true positive. Timestamps which are missed in referece
    are treated as a no-score-region and excluded from computation. Parts of
    this code are inspired by Pascal VOC devkit/ActivityNET.
    
    Parameters
    ----------
    ref : df
        Data frame containing the ground truth instances. Required fields:
        ['file_id', 'Class']
    hyp : df
        Data frame containing the prediction instances. Required fields:
        ['file_id', 'Class', 'llr']
    delta_cp_thresholds : 1darray
        Text/Time Delta Distance Thresholds (>=0)       

    Returns
    -------
    dict 
        Values are tuples [ap, precision, recall]. Keys are Text/Time Delta Distance Thresholds.
        - **ap** (float)
            Average precision score.
        - **precision** (1darray)
            Precision values
        - **recall** (1darray)
            Recall values
    """
        
    # REF has same amount of !score_regions for all runs, which need to be
    # excluded from overall REF count.
    npos = len(ref.loc[ref.Class != 'NO_SCORE_REGION'])    
    output, out = {}, []

    # No Class found.
    if hyp.empty:
        for iout in delta_cp_thresholds:
            output[iout] = 0.0, [0.0, 0.0], [0.0, 1.0]
        return output
 
    # Compute IoU for all hyps incl. NO_SCORE_REGION
    for idx, myhyp in hyp.iterrows():
        out.append(compute_cps(myhyp, ref))
    ihyp = pd.concat(out)

    # Exclude NO_SCORE_REGIONs but keep FP NA's
    ihyp = ihyp.loc[(ihyp.Class_ref != 'NO_SCORE_REGION') | ihyp.Class_ref.isna()]        

    # Sort by confidence score
    ihyp.sort_values(["llr"], ascending=False, inplace=True)        
    ihyp.reset_index(inplace=True, drop=True)        
        
    # Determine TP/FP @ IoU-Threshold
    for iout in delta_cp_thresholds:        
        ihyp[['tp', 'fp']] = [ 0, 1 ]        
        ihyp.loc[~ihyp['Class_ref'].isna() & (ihyp['delta_cp'] <= iout), ['tp', 'fp']] = [ 1, 0 ]
        # Mark TP as FP for duplicate ref matches at lower CS
        nhyp = ihyp.duplicated(subset = ['file_id', 'Class_ref', 'tp'], keep='first')
        ihyp.loc[ihyp.loc[nhyp == True].index, ['tp', 'fp']] = [ 0, 1 ]     
        tp = np.cumsum(ihyp.tp).astype(float)
        fp = np.cumsum(ihyp.fp).astype(float)                  
        rec = (tp / npos).values
        prec = (tp / (tp + fp)).values
        output[iout] = ap_interp(prec, rec), prec, rec 
    return output


def compute_multiclass_cp_pr(ref, hyp, delta_cp_text_thresholds = 100, delta_cp_time_thresholds = 10, nb_jobs=-1):

    """ 
    Compute average precision score (AP) and precision-recall curves for
    each class at a set of specific text/time delta distance thresholds. 
    If references have empty class they will be marked as
    'NO_SCORE_REGION' and are excluded from scoring. 
    
    Parameters
    ----------
    ref: df
        Data frame containing the ground truth instances. Required fields:
        ['file_id', 'Class', 'start', 'end'] 
    hyp: df
        Data frame containing the prediction instances. Required fields:
        ['file_id', 'Class', 'start', 'end', 'llr']
    delta_cp_text_thresholds : 1darray
        Text Delta Distance Thresholds (>=0)
    delta_cp_time_thresholds : 1darray
        Time Delta Distance Thresholds (>=0)
    nb_jobs: int
        Speed up using multi-processing. (-1 use one cpu, 0 use all cpu, N use n
        cpu)

    Returns
    -------
    results: dict [ds]
        Dict of Dataframe w/ type,ap,prec,rec columns w/ Delta-Thresholds as
        keys.
    """
    # Initialize
    scores = {}
    [ scores.setdefault(iout, []) for iout in delta_cp_text_thresholds + delta_cp_time_thresholds ]
    
    # # Iterate over all Classes treating them as a binary detection
    alist = ref.loc[ref.Class != 'NO_SCORE_REGION'].type.unique()

    delta_cp_thresholds = {}
    for idx, act in enumerate(alist):
        if act == "text":
            delta_cp_thresholds[act] = delta_cp_text_thresholds
        else:
            delta_cp_thresholds[act] = delta_cp_time_thresholds

    apScores = Parallel(n_jobs=nb_jobs)(delayed(compute_average_precision_cps)(
            ref=ref.loc[(ref.type == act) | (ref.Class == 'NO_SCORE_REGION')].reset_index(drop=True),                        
            hyp=hyp.loc[(hyp.type == act)].reset_index(drop=True),
            delta_cp_thresholds=delta_cp_thresholds[act]) for idx, act in enumerate(alist))

    for idx, act in enumerate(alist):
        for iout in delta_cp_thresholds[act]:            
            scores[iout].append([act, apScores[idx][iout][0], apScores[idx][iout][1], apScores[idx][iout][2]])       
    
    # Build results for all            
    pr_scores = {}
    for iout in delta_cp_text_thresholds + delta_cp_time_thresholds: 
        pr_scores[iout] = pd.DataFrame(scores[iout], columns = ['type', 'ap', 'precision', 'recall'])

    return pr_scores

def write_type_level_scores(output_dir, results, delta_cp_text_thresholds, delta_cp_time_thresholds):

    type_level_scores = pd.DataFrame()
    for cp in delta_cp_text_thresholds + delta_cp_time_thresholds:
        result_cp = results[cp]
        result_cp["Delta_CP"] = cp
        result_cp.rename(columns={'ap': 'Score', 'type': 'Type'}, inplace = True)
        result_cp['Metric'] = 'AP'

        result_cp = result_cp[["Delta_CP", "Type", "Metric", "Score"]]
        type_level_scores = pd.concat([type_level_scores, result_cp])

    type_level_scores_sort = type_level_scores.sort_values(by=['Type'], ascending=True)
    type_level_scores_sort.to_csv(os.path.join(output_dir, "system_scores.csv"), index = None)
   
def score_cp(ref, hyp, delta_cp_text_thresholds, delta_cp_time_thresholds, output_dir, nb_jobs):
    """ Score System output of Changepoint Detection Task
 
    Parameters
    ----------
    ref: df
        Data frame containing the ground truth instances. Required fields:
        ['file_id', 'Class', 'start', 'end'] 
    hyp: df
        Data frame containing the prediction instances. Required fields:
        ['file_id', 'Class', 'start', 'end', 'llr']
    delta_cp_text_thresholds : 1darray
        Text Delta Distance Thresholds (>=0)
    delta_cp_time_thresholds : 1darray
        Time Delta Distance Thresholds (>=0)
    output_dir: str
        Path to a directory (created on demand) for output files    
    nb_jobs: int
        Speed up using multi-processing. (-1 use one cpu, 0 use all cpu, N use n
        cpu)

    Returns
    -------
    Tuple with following values:
        - **pr_iou_scores** (dict of df)
            multi-type pr for all types
    """    
    # Add text/audio/video info to hyp   
    tad_add_noscore_region(ref,hyp)
    hyp_type = add_type_column(ref, hyp)

    if len(hyp) > 0:
        pr_iou_scores = compute_multiclass_cp_pr(ref, hyp_type, delta_cp_text_thresholds, delta_cp_time_thresholds, nb_jobs)
    else:
        alist = ref.loc[ref.Class != 'NO_SCORE_REGION'].type.unique()
        pr_iou_scores = generate_zero_scores_changepoint(alist, delta_cp_text_thresholds, delta_cp_time_thresholds)

    ensure_output_dir(output_dir)
    write_type_level_scores(output_dir, pr_iou_scores, delta_cp_text_thresholds, delta_cp_time_thresholds)