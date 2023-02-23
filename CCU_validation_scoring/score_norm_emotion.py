import os
import logging
import pprint
from re import M
import numpy as np
import pandas as pd
from .utils import *
import pdb

logger = logging.getLogger('SCORING')


def generate_zero_scores_norm_emotion(ref):
    """
    Generate the result when no match was founded
    """
    y = []
    if len(ref) > 0:
        pr_iou_scores = {}
        unique_combo = ref[["Class", "type"]].value_counts().reset_index()
        unique_combo_pruned = unique_combo.loc[unique_combo.Class != 'NO_SCORE_REGION']
        unique_all = ref[["Class"]].value_counts().reset_index()
        unique_all["type"] = "all"
        unique_all_pruned = unique_all.loc[unique_all.Class != 'NO_SCORE_REGION']
        
        combine_combo_pruned = pd.concat([unique_combo_pruned, unique_all_pruned])
        combine_combo_pruned.sort_values(["Class", "type"], inplace=True)

        final_combo_pruned = combine_combo_pruned.reset_index()
        final_combo_pruned = final_combo_pruned[["Class","type"]]

        if len(final_combo_pruned)>0:
            for i in range(len(final_combo_pruned)):
                y.append( [final_combo_pruned.loc[i, "Class"], final_combo_pruned.loc[i, "type"], 0.0, [0.0, 0.0], [0.0, 1.0] ])
        else:
            logger.error("No matching Classes and types found in system output.")
            y.append( [ 'no_macthing_class', 'no_macthing_type', 0.0, [0.0, 0.0], [0.0, 1.0] ]) 
    else:
        logger.error("No reference to score")
        y.append( ["NA", "NA", "NA", "NA", "NA"])
    return pd.DataFrame(y, columns=['Class', 'type', 'ap', 'precision', 'recall'])


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


def compute_ious(row, ref, class_type):
    """
    Compute the ref/hyp matching table
    """
    refs = ref.loc[ ref['file_id'] == row.file_id ].copy()
    print(f"------Compute iOU class_type={class_type} Hyp={row.to_dict()}")
    #print(ref)
    #exit
    # If there are no references for this hypothesis it's IoU is 0/FP
    if len(refs) == 0:
        return pd.DataFrame(data=[[row.Class, row.file_id, np.nan, np.nan, row.start, row.end, row.llr, 0.0, row.status]],
            columns=['Class', 'file_id', 'start_ref', 'end_ref', 'start_hyp', 'end_hyp', 'llr', 'IoU', 'hyp_status'])
    else:        
        refs['IoU'] = segment_iou(row.start, row.end, [refs.start, refs.end])
        if (len(refs.loc[refs.IoU > 0]) > 1) & ("NO_SCORE_REGION" in refs.loc[refs.IoU == refs.IoU.max()].Class.values):
            #If the class of highest iou is no score region, then pick the second highest
            rmax = refs.loc[refs.IoU == refs.loc[refs.Class != "NO_SCORE_REGION"].IoU.max()]
        else:
            rmax_candidate = refs.loc[refs.IoU == refs.IoU.max()]
            rmax = rmax_candidate.loc[rmax_candidate.start == rmax_candidate.start.min()]
        rout = rmax.rename(columns={'start':'start_ref', 'end':'end_ref'})
        rout[['start_hyp', 'end_hyp', 'llr']] = row.start, row.end, row.llr
        if (class_type == "norm"):
            rout['hyp_status'] = row.status
        return rout

def compute_average_precision_tad(ref, hyp, Class, iou_thresholds=[0.2], task=None):
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
    task:
        string that indicates task name. e.g. norm/emotion        

    Returns
    -------
    output: 
        Values are tuples [ap, precision, recall]. Keys are tIoU thr.
        - **ap** (float)
            Average precision score.
        - **precision** (1darray)
            Precision values
        - **recall** (1darray)
            Recall values

    final_alignment_df: instance alignment dataframe
    """
    print(f"\n=========================================================================")
    print(f"=============  compute_average_precision_tad Class={Class} =====================")
    print(ref[ref.Class == Class])
    print(hyp[hyp.Class == Class])
    print(f"=============  compute_average_precision_tad Class={Class} =====================")


    # REF has same amount of !score_regions for all runs, which need to be
    # excluded from overall REF count.
    npos = len(ref.loc[ref.Class.str.contains('NO_SCORE_REGION')==False])    
    output, out = {}, []
    alignment_df = pd.DataFrame()

    # No Class found.
    if hyp.empty:
        for iout in iou_thresholds:
            output[iout] = 0.0, [0.0, 0.0], [0.0, 1.0]
        alignment_df = generate_all_fn_alignment_file(ref, task)
        return output,alignment_df

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)
    
    if (False):
        print("------------Remove Noscores-----------")
        print(ref.loc[ref.Class.str.contains('NO_SCORE_REGION') == True])
        print("------------Residual Noscores-----------")
        ref = ref.loc[ref.Class.str.contains('NO_SCORE_REGION') == False]
        print(ref)
    
    # Compute IoU for all hyps incl. NO_SCORE_REGION
    print("---  Computing_ious  -")
    for idx, myhyp in hyp.iterrows():
        out.append(compute_ious(myhyp, ref, task))
    ihyp = pd.concat(out)
    print("-----------------------")
    print(out)
    #exit(1)
    
    # Capture naked false alarms that have no overlap with anything regardless of if it is a NO_SCORE_REGION
    #ihyp_naked_fa = ihyp.loc[ihyp.IoU == 0.0]
    #print("---- Naked FAs -----")
    #print(ihyp_naked_fa)
    
    # Exclude NO_SCORE_REGIONs but keep FP NA's
    #ihyp = ihyp.loc[(ihyp.Class.str.contains('NO_SCORE_REGION') == False) | ihyp.start_ref.isna()]
    ihyp = ihyp.loc[(ihyp.Class.str.contains('NO_SCORE_REGION') == False) | ((ihyp.Class.str.contains('NO_SCORE_REGION') == True) & (ihyp['IoU'] == 0.0)) | ihyp.start_ref.isna()]
    #print("----- ihyp keep ------------------")
    #print(ihyp)
    #exit(1)
    
    if ihyp.empty:
        for iout in iou_thresholds:
            output[iout] = 0.0, [0.0, 0.0], [0.0, 1.0]
        alignment_df = generate_all_fn_alignment_file(ref, task)
        return output,alignment_df

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
        ihyp.sort_values(["llr", "file_id"], ascending=[False, True], inplace=True)
        #print("------------PRe Alignment--------------");
        #print(ihyp)
 
        ### Update the data for naked false alarms
        ihyp.loc[ihyp.loc[ihyp['IoU'] == 0.0].index, ['Class', 'tp', 'fp', 'start_ref', 'end_ref']] = [ Class, 0, 1 , -1, -1]
       
        tp = np.cumsum(ihyp.tp).astype(float)
        fp = np.cumsum(ihyp.fp).astype(float)

        # after filtering 
        ihyp["cum_tp"] = tp
        ihyp["cum_fp"] = fp

        fhyp = ihyp
        thyp = fhyp.duplicated(subset = ['llr'], keep='last')
        fhyp = fhyp.loc[thyp == False]
                     
        rec = (np.array(fhyp["cum_tp"]) / npos)
        prec = (np.array(fhyp["cum_tp"]) / (np.array(fhyp["cum_tp"]) + np.array(fhyp["cum_fp"])))

        output[iout] = ap_interp(prec, rec), prec, rec

        ihyp_fields = ["Class","type","tp","fp","file_id","start_ref","end_ref","start_hyp","end_hyp","IoU","llr"]
        if (task == "norm"):
            ihyp_fields.append("status")
            ihyp_fields.append("hyp_status")
        ihyp = ihyp[ihyp_fields]
        
        alignment_df = pd.concat([alignment_df, ihyp])
        #print("-------------------------Alignment_df--------------");
        #print(alignment_df)


        
    final_alignment_df = generate_alignment_file(ref.loc[ref.Class.str.contains('NO_SCORE_REGION')==False], alignment_df, task)

    return output,final_alignment_df


def compute_multiclass_iou_pr(ref, hyp, iou_thresholds=0.2, mapping_df = None, class_type = None):
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
    mapping_df:
        norm mapping dataframe
    class_type:
        string that indicates task name. e.g. norm/emotion

    Returns
    -------
    pr_scores: dict [ds]
        Dict of Dataframe w/ class,ap,prec,rec columns w/ IoU-Thresholds as
        keys.

    final_alignment_df: instance alignment dataframe
    """
    #print(f"\n**************************************************************************")
    #print(f"************ compute_multiclass_iou_pr ioU_thresdholdd={iou_thresholds} ******************")
    #print("Ref")
    #print(ref)
    #print("Hyp")
    #print(hyp)
    #print(f"*************************************************************************")

    # Initialize
    scores = {}
    [ scores.setdefault(iout, []) for iout in iou_thresholds ]
    
    # Iterate over all Classes treating them as a binary detection
    unique_combo = ref[["Class", "type"]].value_counts().reset_index()
    unique_combo_pruned = unique_combo.loc[unique_combo.Class != 'NO_SCORE_REGION']
    unique_all = ref[["Class"]].value_counts().reset_index()
    unique_all["type"] = "all"
    unique_all_pruned = unique_all.loc[unique_all.Class != 'NO_SCORE_REGION']
    
    combine_combo_pruned = pd.concat([unique_combo_pruned, unique_all_pruned])
    combine_combo_pruned.sort_values(["Class", "type"], inplace=True)

    final_combo_pruned = combine_combo_pruned.reset_index()
    final_combo_pruned = final_combo_pruned[["Class","type"]]  

    apScores = []
    alignment_df = pd.DataFrame()
    for i in range(len(final_combo_pruned)):

        if final_combo_pruned.loc[i, "type"] == "all":
            match_type = ["audio","text","video"]
        else:
            match_type = [final_combo_pruned.loc[i, "type"]]

        if mapping_df is not None:
            sub_mapping_df = mapping_df.loc[mapping_df.ref_norm == final_combo_pruned.loc[i, "Class"]]
            if not sub_mapping_df.empty:
                final_sub_hyp = replace_hyp_norm_mapping(sub_mapping_df, hyp, final_combo_pruned.loc[i, "Class"])
                final_sub_hyp_type = final_sub_hyp.loc[(final_sub_hyp.type.isin(match_type))]
            else:
                final_sub_hyp_type = hyp.loc[(hyp.Class == final_combo_pruned.loc[i, "Class"]) & (hyp.type.isin(match_type))]

            hyp_scoring = final_sub_hyp_type.reset_index(drop=True)
        
        else:
            hyp_scoring = hyp.loc[(hyp.Class == final_combo_pruned.loc[i, "Class"]) & (hyp.type.isin(match_type))].reset_index(drop=True)

        apScore, alignment = compute_average_precision_tad(
                ref=ref.loc[((ref.Class == final_combo_pruned.loc[i, "Class"]) | (ref.Class == 'NO_SCORE_REGION')) & (ref.type.isin(match_type))].reset_index(drop=True),                        
                hyp=hyp_scoring,
                Class=final_combo_pruned.loc[i, "Class"],
                iou_thresholds=iou_thresholds,
                task=class_type)
                    
        apScores.append(apScore)
        if final_combo_pruned.loc[i, "type"] == "all":
            alignment_df = pd.concat([alignment_df, alignment])

    final_alignment_df = alignment_df.drop_duplicates()

    for i in range(len(final_combo_pruned)):
        for iout in iou_thresholds:            
            scores[iout].append([final_combo_pruned.loc[i, "Class"], final_combo_pruned.loc[i, "type"], apScores[i][iout][0], apScores[i][iout][1], apScores[i][iout][2]])

    # Build results for all            
    pr_scores = {}
    for iout in iou_thresholds: 
        pr_scores[iout] = pd.DataFrame(scores[iout], columns = ['Class', 'type', 'ap', 'precision', 'recall'])

    return pr_scores, final_alignment_df

def sumup_tad_system_level_scores(pr_iou_scores, iou_thresholds, class_type, output_dir):
    """
    Write aggregate result into a file
    """
    map_scores_threshold = pd.DataFrame()
    for iout in sorted(iou_thresholds):
        pr_scores = pr_iou_scores[iout]
        if pr_scores["ap"].values[0] != "NA":
            map_scores = pr_scores.groupby('type')['ap'].mean().reset_index()
            map_scores.ap = map_scores.ap.round(3)
            
            map_scores = map_scores.rename(columns={'type': 'genre', 'ap': 'value'})
        else:
            map_scores = pd.DataFrame([["NA","NA"]], columns=["genre","value"])
        
        map_scores["correctness_criteria"] = "{iou=%s}" % iout
        map_scores["metric"] = "mAP"

        if class_type == "norm":
            map_scores["task"] = "nd"
        if class_type == "emotion":
            map_scores["task"] = "ed"
        
        map_scores = map_scores[["task","genre","metric","value","correctness_criteria"]]
        map_scores_threshold = pd.concat([map_scores, map_scores_threshold])
    
    map_scores_threshold.to_csv(os.path.join(output_dir, "scores_aggregated.tab"), sep = "\t", index = None)
        
def sumup_tad_class_level_scores(pr_iou_scores, iou_thresholds, output_dir):
    """
    Write class level result into a file
    """
    prs_threshold = pd.DataFrame()   
    for iout in sorted(iou_thresholds):        
        prs = pr_iou_scores[iout]
        if prs["ap"].values[0] != "NA":
            prs.ap = prs.ap.round(3)
        prs["metric"] = "AP"        
        prs["correctness_criteria"] = "{iou=%s}" % iout
        prs = prs.rename(columns={'Class': 'class', 'type': 'genre', 'ap': 'value'})
        prs = prs[["class","genre","metric","value","correctness_criteria"]]
        prs_threshold = pd.concat([prs, prs_threshold])
    
    prs_threshold.to_csv(os.path.join(output_dir, "scores_by_class.tab"), sep = "\t", index = None)
   
def score_tad(ref, hyp, class_type, iou_thresholds, output_dir, mapping_df):
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
    output_dir: str
        Path to a directory (created on demand) for output files   
    mapping_df:
        norm mapping dataframe 
    """    

    # FIXME: Use a No score-region parameter
    tad_add_noscore_region(ref,hyp)

    if len(ref) > 0:
        if len(hyp) > 0:
            pr_iou_scores, final_alignment_df = compute_multiclass_iou_pr(ref, hyp, iou_thresholds, mapping_df, class_type)
        else:
            pr_iou_scores = {}
            for iout in iou_thresholds:
                pr_iou_scores[iout] = generate_zero_scores_norm_emotion(ref)
            final_alignment_df = generate_all_fn_alignment_file(ref, class_type)
    else:
        pr_iou_scores = {}
        for iout in iou_thresholds:
            pr_iou_scores[iout] = generate_zero_scores_norm_emotion(ref)
        final_alignment_df = pd.DataFrame([["NA","NA","NA","NA","NA","NA","NA","NA", (["",""] if (class_type == "norm") else [])]],
                                          columns=["class", "file_id", "eval", "ref", "sys", "llr", "parameters","sort",(["ref_status","hyp_status"] if (class_type == "norm") else [])])

    ensure_output_dir(output_dir)
    final_alignment_df_sorted = final_alignment_df.sort_values(by=['class', 'file_id', 'sort'])
    final_alignment_df_sorted.to_csv(os.path.join(output_dir, "instance_alignment.tab"), index = False, quoting=3, sep="\t", escapechar="\t",
                                     columns = ["class","file_id","eval","ref","sys","llr","parameters"] + (["ref_status","hyp_status"] if (class_type == "norm") else []))
    sumup_tad_system_level_scores(pr_iou_scores, iou_thresholds, class_type, output_dir)
    sumup_tad_class_level_scores(pr_iou_scores, iou_thresholds, output_dir)


