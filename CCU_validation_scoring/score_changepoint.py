import os
import logging
import numpy as np
import pandas as pd
from .utils import *
import json

logger = logging.getLogger('SCORING')


def generate_zero_scores_changepoint(ref, delta_cp_text_thresholds, delta_cp_time_thresholds):
    """
    Generate the result when no match was founded
    """
    if len(ref) > 0:
        labels = ref.loc[ref.Class != 'NO_SCORE_REGION'].type.unique()
        if len(labels)>0:
            y_text = []
            y_time = []
            for idx, act in enumerate(labels):
                if act == "text":
                    y_text.append( ["cp", act, 0.0, [0.0], [0.0], [0.0] ])
                else:
                    y_time.append( ["cp", act, 0.0, [0.0], [0.0], [0.0] ])
            df_text = pd.DataFrame(y_text, columns=['Class', 'type', 'ap', 'precision', 'recall'])
            df_time = pd.DataFrame(y_time, columns=['Class', 'type', 'ap', 'precision', 'recall'])
            pr_iou_scores = {}
            for iout in delta_cp_text_thresholds:
                pr_iou_scores[iout] = df_text
            for iout in delta_cp_time_thresholds:
                pr_iou_scores[iout] = df_time

        else:
            y = []
            logger.error("No matching Types found in system output.")
            y.append( [ "cp", 'no_macthing_Type', 0.0, [0.0], [0.0], [0.0] ])
            df_whole = pd.DataFrame(y, columns=['Class', 'type', 'ap', 'precision', 'recall'])
            pr_iou_scores = {}
            for iout in delta_cp_text_thresholds + delta_cp_time_thresholds:
                pr_iou_scores[iout] = df_whole
    
    else:
        y = []
        logger.error("No reference to score")
        y.append( [ 'NA', 'NA', 'NA', 'NA', 'NA', 'NA' ])
        df_whole = pd.DataFrame(y, columns=['Class', 'type', 'ap', 'precision', 'recall', 'llr'])
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
    """
    Compute the ref/hyp matching table
    """
    refs = ref.loc[ ref['file_id'] == row.file_id ].copy()    
    # If there are no references for this hypothesis it's IoU is 0/FP
    if len(refs) == 0:
        return pd.DataFrame(data=[[row.type, row.file_id, np.nan, row.Class, row.llr, 0.0]],
            columns=['type', 'file_id', 'Class_ref', 'Class_hyp', 'llr', 'delta_cp'])
    else:        
        refs['delta_cp'] = segment_cp(row.Class, [refs.Class])
        rmin_candidate = refs.loc[refs.delta_cp == refs.delta_cp.min()]
        rmin = rmin_candidate.loc[rmin_candidate.Class == rmin_candidate.Class.min()]
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
    output:
        Values are tuples [ap, precision, recall]. Keys are Text/Time Delta Distance Thresholds.
        - **ap** (float)
            Average precision score.
        - **precision** (1darray)
            Precision values
        - **recall** (1darray)
            Recall values

    final_alignment_df: instance alignment dataframe
    """
    # REF has same amount of !score_regions for all runs, which need to be
    # excluded from overall REF count.
    npos = len(ref.loc[ref.Class != 'NO_SCORE_REGION'])    
    output, out = {}, []
    alignment_df = pd.DataFrame()

    # No Class found.
    if hyp.empty:
        for iout in delta_cp_thresholds:
            output[iout] = 0.0, [0.0], [0.0], [0.0]
        alignment_df = generate_all_fn_alignment_file(ref, "changepoint")
        return output,alignment_df
    
    # Compute IoU for all hyps incl. NO_SCORE_REGION
    for idx, myhyp in hyp.iterrows():
        out.append(compute_cps(myhyp, ref))
    ihyp = pd.concat(out)
    
    # Exclude NO_SCORE_REGIONs but keep FP NA's
    ihyp = ihyp.loc[(ihyp.Class_ref != 'NO_SCORE_REGION') | ihyp.Class_ref.isna()]

    if ihyp.empty:
        for iout in delta_cp_thresholds:
            output[iout] = 0.0, [0.0], [0.0], [0.0]
        alignment_df = generate_all_fn_alignment_file(ref, "changepoint")
        return output,alignment_df

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
        ihyp.sort_values(["llr", "file_id"], ascending=[False, True], inplace=True)
        tp = np.cumsum(ihyp.tp).astype(float)
        fp = np.cumsum(ihyp.fp).astype(float)
                  
        # after filtering 
        ihyp["cum_tp"] = tp
        ihyp["cum_fp"] = fp

        fhyp = ihyp
        thyp = fhyp.duplicated(subset = ['llr'], keep='last')
        fhyp = fhyp.loc[thyp == False]
                     
        llr = np.array(fhyp["llr"])
        rec = (np.array(fhyp["cum_tp"]) / npos)
        prec = (np.array(fhyp["cum_tp"]) / (np.array(fhyp["cum_tp"]) + np.array(fhyp["cum_fp"])))

        output[iout] = ap_interp(prec, rec), prec, rec, llr

        ihyp = ihyp[["tp","fp","file_id","type","Class_ref","Class_hyp","delta_cp","llr"]]
        alignment_df = pd.concat([alignment_df, ihyp])
    final_alignment_df = generate_alignment_file(ref.loc[ref.Class != 'NO_SCORE_REGION'], alignment_df, "changepoint")

    return output,final_alignment_df


def compute_multiclass_cp_pr(ref, hyp, delta_cp_text_thresholds = 100, delta_cp_time_thresholds = 10):

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

    Returns
    -------
    pr_scores: dict [ds]
        Dict of Dataframe w/ type,ap,prec,rec columns w/ Delta-Thresholds as
        keys.

    final_alignment_df: instance alignment dataframe
    """
    # Initialize

    scores = {}
    #[ scores.setdefault(iout, pd.DataFrame([], columns = ['type', 'ap', 'precision', 'recall', 'llr'])) for iout in delta_cp_text_thresholds + delta_cp_time_thresholds ]
    from collections import defaultdict
    scores = defaultdict(list)

    ### Capture the noscores for later use
    ref_noscore = ref.loc[ref.impact_scalar == 'NO_SCORE_REGION']
    ref = ref.loc[ref.impact_scalar != 'NO_SCORE_REGION']
    #print("orig")
    #print("ref_noscore")
    #print(ref_noscore)
    #print("ref")
    #print(ref)    
    #print("hyp")
    #print(hyp)     
    
    ### Remove system detects in the NOSCORES
    for index, row in ref_noscore.iterrows():
        #print("Filter Row {} st={} en={}".format(row.file_id, row.start, row.end))
        f, s, e = [row.file_id, row.start, row.end]
        hyp.drop(hyp[(hyp.file_id == f) & (s <= hyp.Class) & (hyp.Class <= e)].index, inplace=True)
        

    #print("post filter hyp")
    #print(hyp)

    # # Iterate over all Classes treating them as a binary detection
    alist = ref.loc[ref.Class != 'NO_SCORE_REGION'].type.unique()

    delta_cp_thresholds = {"text": delta_cp_text_thresholds, "video": delta_cp_time_thresholds, "audio": delta_cp_time_thresholds}    
    apScores = []
    alignment_df = pd.DataFrame()
    for act in alist:
        
        apScore, alignment = compute_average_precision_cps(
                ref=ref.loc[(ref.type == act) | (ref.Class == 'NO_SCORE_REGION')].reset_index(drop=True),                        
                hyp=hyp.loc[(hyp.type == act)].reset_index(drop=True),
                delta_cp_thresholds=delta_cp_thresholds[act])

        apScores.append(apScore)
        alignment_df = pd.concat([alignment_df, alignment])

    final_alignment_df = alignment_df.drop_duplicates()

    for idx, act in enumerate(alist):
        for iout in delta_cp_thresholds[act]:
            scores[iout].append([act, apScores[idx][iout][0], apScores[idx][iout][1], apScores[idx][iout][2], apScores[idx][iout][3]])       
    
    # Build results for all            
    pr_scores = {}
    for iout in delta_cp_text_thresholds + delta_cp_time_thresholds: 
        pr_scores[iout] = pd.DataFrame(scores[iout], columns = ['type', 'ap', 'precision', 'recall', 'llr'])

    return pr_scores, final_alignment_df

def write_type_level_scores(output_dir, results, delta_cp_text_thresholds, delta_cp_time_thresholds):
    """
    Write class level result into a file.  THIS UPDATES THE SOURCE DATA FRAME
    """
    results_long = pd.DataFrame()
    type_level_scores = pd.DataFrame()
    for cp in delta_cp_text_thresholds + delta_cp_time_thresholds:
        result_cp = results[cp]
        result_cp["correctness_criteria"] = "{delta_cp=" + str(cp) + "}"
        result_cp.rename(columns={'ap': 'value', 'type': 'genre'}, inplace = True)
        result_cp['metric'] = 'AP'
        result_cp["class"] = 'cp'

        if (len(result_cp.index) > 0):
            ### Make the long form!!!
            ### AP
            temp = results[cp].copy(deep=True)
            temp['metric'] = "AP"
            temp = temp[["class","genre","metric","value","correctness_criteria"]]
            results_long = pd.concat([results_long, temp]);
            ### PRCurve
            temp = results[cp].copy(deep=True)
            temp['metric'] = "PRCurve_json"
            d = {"precision": [ x for x in temp['precision'].tolist()[0] ],
                 'recall': [x for x in temp['recall'].tolist()[0] ],
                 'llr': [x for x in temp['llr'].tolist()[0] ]}
            temp['value'] = json.dumps(d)
        
            temp = temp[["class","genre","metric","value","correctness_criteria"]]
            results_long = pd.concat([results_long, temp]);

    results_long.to_csv(os.path.join(output_dir, "scores_by_class.tab"), sep = "\t", index = None, quoting=None)   
   
def score_cp(ref, hyp, delta_cp_text_thresholds, delta_cp_time_thresholds, output_dir):
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
    """    
    # Add text/audio/video info to hyp   
    tad_add_noscore_region(ref,hyp)

    if len(ref) > 0:
        if len(hyp) > 0:
            pr_iou_scores, final_alignment_df = compute_multiclass_cp_pr(ref, hyp, delta_cp_text_thresholds, delta_cp_time_thresholds)
        else:
            pr_iou_scores = generate_zero_scores_changepoint(ref, delta_cp_text_thresholds, delta_cp_time_thresholds)
            final_alignment_df = generate_all_fn_alignment_file(ref, "changepoint")
    else:
        pr_iou_scores = generate_zero_scores_changepoint(ref, delta_cp_text_thresholds, delta_cp_time_thresholds)
        final_alignment_df = pd.DataFrame([["cp","NA","NA","NA","NA","NA","NA"]], columns=["class", "file_id", "eval", "ref", "sys", "llr", "parameters"])

    ### Hack the pr_iou_scores structure.  For ED and ND, genre is in the type column so replicate the genre column here
        
    ensure_output_dir(output_dir)
    final_alignment_df_sorted = final_alignment_df.sort_values(by=['class', 'file_id', 'sys', 'ref'])
    final_alignment_df_sorted.to_csv(os.path.join(output_dir, "instance_alignment.tab"), index = False, quoting=3, sep="\t", escapechar="\t")
    write_type_level_scores(output_dir, pr_iou_scores, delta_cp_text_thresholds, delta_cp_time_thresholds)    
    for iou, class_data in pr_iou_scores.items():
        pr_iou_scores[iou]['type'] = pr_iou_scores[iou]['genre']
        pr_iou_scores[iou]['Class'] = pr_iou_scores[iou]['class']
        pr_iou_scores[iou]['llr'] = pr_iou_scores[iou]['llr']

    graph_info_dict = []
    generate_alignment_statistics(final_alignment_df_sorted, "cd", output_dir = output_dir, info_dict = graph_info_dict)
    make_pr_curve_for_cd(pr_iou_scores, "cd", "cd", output_dir, info_dict = graph_info_dict)
    graph_info_df = pd.DataFrame(graph_info_dict)
    graph_info_df.to_csv(os.path.join(output_dir, "graph_info.tab"), index = False, quoting=3, sep="\t", escapechar="\t")
    
