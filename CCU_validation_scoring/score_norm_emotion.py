import os
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .utils import *

logger = logging.getLogger('SCORING')


def generate_zero_scores_norm_emotion(labels):
    y = []
    if len(labels)>0:
        for i in labels:
            y.append( [ i, 0.0, [0.0, 0.0], [0.0, 1.0] ])
    else:
        logger.error("No matching Classes found in system output.")
        y.append( [ 'no_macthing_Class', 0.0, [0.0, 0.0], [0.0, 1.0] ]) 
    return pd.DataFrame(y, columns=['Class', 'ap', 'precision', 'recall'])


def segment_iou(ref_start, ref_end, tgts):
    """
    Compute __Temporal__ Intersection over Union (__IoU__) as defined in 
    [1], Eq.(1) given start and endpoints of intervals __g__ and __p__.    
    Vectorized impl. from ActivityNET/Pascal VOC devkit.

    Parameters
    ----------
    ref_start: float
        starting frame of source segement
    ref_end: float
        end frame of source segement        
    tgts : 2d array
        Temporal test segments containing [starting x N, ending X N] times.

    Returns
    -------
    tIoU : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(ref_start, tgts[0])
    tt2 = np.minimum(ref_end, tgts[1])    
    # Segment intersection including Non-negative overlap score
    inter = (tt2 - tt1).clip(0)    
    # Segment union.
    union = (tgts[1] - tgts[0]) + (ref_end - ref_start) - inter    
    tIoU = inter.astype(float) / union
    return tIoU


def compute_ious(row, ref):
    refs = ref.loc[ ref['file_id'] == row.file_id ].copy()    
    # If there are no references for this hypothesis it's IoU is 0/FP
    if len(refs) == 0:
        return pd.DataFrame(data=[[row.Class, row.file_id, np.nan, np.nan, row.start, row.end, row.llr, 0.0]],
            columns=['Class', 'file_id', 'start_ref', 'end_ref', 'start_hyp', 'end_hyp', 'llr', 'IoU'])
    else:        
        refs['IoU'] = segment_iou(row.start, row.end, [refs.start, refs.end])
        rmax = refs.loc[refs.IoU == refs.IoU.max()]
        rout = rmax.rename(columns={'start':'start_ref', 'end':'end_ref'})
        rout[['start_hyp', 'end_hyp', 'llr']] = row.start, row.end, row.llr
        return rout

def compute_average_precision_tad(ref, hyp, iou_thresholds=[0.2]):
    """ 
    Compute average precision and precision-recall curve at specific IoU
    thresholds between ground truth and predictions data frames. If multiple
    predictions occur for the same predicted segment, only the one with highest
    tIoU is matched as true positive. Classes which are missed in referece
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
    iou_thresholds : 1darray, optional
        Temporal IoU Threshold (>=0)        

    Returns
    -------
    dict 
        Values are tuples [ap, precision, recall]. Keys are tIoU thr.
        - **ap** (float)
            Average precision score.
        - **precision** (1darray)
            Precision values
        - **recall** (1darray)
            Recall values
    """
        
    # REF has same amount of !score_regions for all runs, which need to be
    # excluded from overall REF count.
    npos = len(ref.loc[ref.Class.str.contains('NO_SCORE_REGION')==False])    
    output, out = {}, []

    # No Class found.
    if hyp.empty:
        for iout in iou_thresholds:
            output[iout] = 0.0, [0.0, 0.0], [0.0, 1.0]
        return output

    # Compute IoU for all hyps incl. NO_SCORE_REGION
    for idx, myhyp in hyp.iterrows():
        out.append(compute_ious(myhyp, ref))
    ihyp = pd.concat(out)

    # Exclude NO_SCORE_REGIONs but keep FP NA's
    ihyp = ihyp.loc[(ihyp.Class.str.contains('NO_SCORE_REGION') == False) | ihyp.start_ref.isna()]        

    # Sort by confidence score
    ihyp.sort_values(["llr"], ascending=False, inplace=True)        
    ihyp.reset_index(inplace=True, drop=True)
         
    # Determine TP/FP @ IoU-Threshold
    for iout in iou_thresholds:        
        ihyp[['tp', 'fp']] = [ 0, 1 ]        
        ihyp.loc[~ihyp['start_ref'].isna() & (ihyp['IoU'] >= iout), ['tp', 'fp']] = [ 1, 0 ]
        # Mark TP as FP for duplicate ref matches at lower CS
        nhyp = ihyp.duplicated(subset = ['file_id', 'start_ref', 'end_ref', 'tp'], keep='first')
        ihyp.loc[ihyp.loc[nhyp == True].index, ['tp', 'fp']] = [ 0, 1 ]        
        tp = np.cumsum(ihyp.tp).astype(float)
        fp = np.cumsum(ihyp.fp).astype(float)                      
        rec = (tp / npos).values
        prec = (tp / (tp + fp)).values
        output[iout] = ap_interp(prec, rec), prec, rec   
    return output


def compute_multiclass_iou_pr(ref, hyp, iou_thresholds=0.2, nb_jobs=-1, mapping_df = None):
    """ Compute average precision score (AP) and precision-recall curves for
    each class at a set of specific temp. intersection-over-union (tIoU)
    thresholds. If references have empty class they will be marked as
    'NO_SCORE_REGION' and are excluded from scoring. 
    
    Parameters
    ----------
    ref: df
        Data frame containing the ground truth instances. Required fields:
        ['file_id', 'Class', 'start', 'end'] 
    hyp: df
        Data frame containing the prediction instances. Required fields:
        ['file_id', 'Class', 'start', 'end', 'llr']
    iou_thresholds: 1darray
        List of IoU levels to score at.
    nb_jobs: int
        Speed up using multi-processing. (-1 use one cpu, 0 use all cpu, N use n
        cpu)

    Returns
    -------
    results: dict [ds]
        Dict of Dataframe w/ class,ap,prec,rec columns w/ IoU-Thresholds as
        keys.
    """
    # Initialize
    scores = {}
    [ scores.setdefault(iout, []) for iout in iou_thresholds ]
    
    # Iterate over all Classes treating them as a binary detection
    alist = ref.loc[ref.Class.str.contains('NO_SCORE_REGION')==False].Class.unique()        

    if mapping_df is not None:
        apScores = []
        for idx, act in enumerate(alist):
            sub_mapping_df = mapping_df.loc[mapping_df.ref_norm == act]
            if not sub_mapping_df.empty:
                final_sub_hyp = replace_hyp_norm_mapping(sub_mapping_df, hyp, act)
            else:
                final_sub_hyp = hyp.loc[(hyp.Class == act)]

            apScore = compute_average_precision_tad(
                    ref=ref.loc[(ref.Class == act) | (ref.Class == 'NO_SCORE_REGION')].reset_index(drop=True),                        
                    hyp=final_sub_hyp.reset_index(drop=True),
                    iou_thresholds=iou_thresholds)
                        
            apScores.append(apScore)

    else:         
        apScores = Parallel(n_jobs=nb_jobs)(delayed(compute_average_precision_tad)(
                ref=ref.loc[(ref.Class == act) | (ref.Class == 'NO_SCORE_REGION')].reset_index(drop=True),                        
                hyp=hyp.loc[(hyp.Class == act)].reset_index(drop=True),
                iou_thresholds=iou_thresholds) for idx, act in enumerate(alist))

    for idx, act in enumerate(alist):
        for iout in iou_thresholds:            
            scores[iout].append([act, apScores[idx][iout][0], apScores[idx][iout][1], apScores[idx][iout][2]])        
        
    # Build results for all            
    pr_scores = {}
    for iout in iou_thresholds: 
        pr_scores[iout] = pd.DataFrame(scores[iout], columns = ['Class', 'ap', 'precision', 'recall'])

    return pr_scores

def _sumup_tad_system_level_scores(metrics, pr_iou_scores, iou_thresholds):
    """ Map internal to public representation. """
    ciou = {}
    for iout in iou_thresholds:
        pr_scores = pr_iou_scores[iout]
        co = {}
        if 'map'         in metrics: co['mAP']        = round(np.mean(pr_scores.ap), 4)        
        ciou[iout] = co
    return ciou
        
def _sumup_tad_class_level_scores(metrics, pr_iou_scores, iou_thresholds):
    """ Map internal to public representation. Scores per Class and IoU Level """  
    act = {}    
    for iout in iou_thresholds:        
        prs = pr_iou_scores[iout]        
        for index, row in prs.iterrows():            
            co = {}
            if 'map'         in metrics: co[        "AP"] = round(row['ap'], 4)
            Class = row['Class']
            if Class not in act.keys():
                act[Class] = {}
            act[Class][iout] = co
    return act

def write_system_level_scores(output_dir, results):

    ious = []
    metrics = []
    scores = []

    for iou, values in results.items():
        for metric, value in values.items():
            ious.append(iou)
            metrics.append(metric)
            scores.append(value)

    system_level_scores = pd.DataFrame({"IoU": ious, "Metric": metrics, "Score": scores})
    system_level_scores.to_csv(os.path.join(output_dir, "system_scores.csv"), index = None)


def write_class_level_scores(output_dir, results, class_type):

    classes = []
    ious = []
    metrics = []
    scores = []

    for Class, values in results.items():
        for iou, value in values.items():
            for metric, score in value.items():
                classes.append(Class)
                ious.append(iou)
                metrics.append(metric)
                scores.append(score)

    class_level_scores = pd.DataFrame({"Class": classes, "IoU": ious, "Metric": metrics, "Score": scores})
    class_level_scores["Class_type"] = class_type
    class_level_scores = class_level_scores[["Class_type", "Class", "IoU", "Metric", "Score"]]
    class_level_scores.to_csv(os.path.join(output_dir, "class_scores.csv"), index = None)
   
def score_tad(ref, hyp, class_type, iou_thresholds, metrics, output_dir, nb_jobs, mapping_df):
    """ Score System output of Norm/Emotion Detection Task
 
    Parameters
    ----------
    ref: df
        Data frame containing the ground truth instances. Required fields:
        ['file_id', 'Class', 'start', 'end'] 
    hyp: df
        Data frame containing the prediction instances. Required fields:
        ['file_id', 'Class', 'start', 'end', 'llr']
    class_type:
        string that indicates task name. e.g. norm/emotion
    iou_thresholds: 1darray [int]
        List of IoU levels to score at.
    metrics: list[str] 
        Array of metrics to include
    output_dir: str
        Path to a directory (created on demand) for output files    
    nb_jobs: int
        Speed up using multi-processing. (-1 use one cpu, 0 use all cpu, N use n
        cpu)

    Returns
    -------
    Tuple with following values:
        - **pr_iou_scores** (dict of df)
            multi-class pr for all classes and ious
        - **results** (df)
            metrics for system level
        - **al_results** (df)
            metrics for class level
    """    

    # FIXME: Use a No score-region parameter
    tad_add_noscore_region(ref,hyp)
    # Fix out of scope and NA's
    remove_out_of_scope_activities(ref,hyp,class_type) 
    
    if len(hyp) > 0:
        pr_iou_scores = compute_multiclass_iou_pr(ref, hyp, iou_thresholds, nb_jobs, mapping_df)
    else:
        pr_iou_scores = {}
        alist = ref.loc[ref.Class.str.contains('NO_SCORE_REGION')==False].Class.unique()
        [ pr_iou_scores.setdefault(iout, generate_zero_scores_norm_emotion(alist)) for iout in iou_thresholds ]

    results = _sumup_tad_system_level_scores(metrics, pr_iou_scores, iou_thresholds)
    al_results = _sumup_tad_class_level_scores(metrics, pr_iou_scores, iou_thresholds)

    ensure_output_dir(output_dir)
    write_system_level_scores(output_dir, results)
    write_class_level_scores(output_dir, al_results, class_type)


