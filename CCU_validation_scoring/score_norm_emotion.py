import os
import re
import pprint
import logging
import pprint
from re import M
import numpy as np
import pandas as pd
from .utils import *
from .aggregate import *
import pdb
import matplotlib.pyplot as plt
import json

logger = logging.getLogger('SCORING')

def f1(precision, recall):
    if (precision + recall == 0):
        return(float("nan"))
    return(2 * (precision * recall) / (precision + recall))

def generate_zero_scores_norm_emotion(ref):
    """
    Generate the result when no match was founded
    """ 
    empty =  { 'AP': 0.0,
               'prcurve:precision': None,
               'prcurve:recall': None,
               'prcurve:llr': None,
               'precision_at_MinLLR': None,
               'recall_at_MinLLR': None,
               'f1_at_MinLLR': None,
               'llr_at_MinLLR': None,
               'sum_tp_at_MinLLR': None,
               'sum_fp_at_MinLLR': None,
               'sum_scaled_tp_at_MinLLR': None, 
               'sum_scaled_fp_at_MinLLR': None,
               'scaled_recall_at_MinLLR': None,
               'scaled_precision_at_MinLLR': None,
               'scaled_f1_at_MinLLR': None
             }    
    if (ref is None):
        return(empty.copy())
    
    empty['Class'] = 'no_matching_class'
    empty['type'] = 'no_matching_type'

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
                t = empty.copy()
                t['Class'] = final_combo_pruned.loc[i, "Class"]
                t['type'] =  final_combo_pruned.loc[i, "type"]
                y.append(t)
        else:
            logger.error("No matching Classes and types found in system output.")
            y.append(empty.copy)
    else:
        logger.error("No reference to score")
        y.append(empty.copy)
    return(y)


    # y = []
    # if len(ref) > 0:
    #     pr_iou_scores = {}
    #     unique_combo = ref[["Class", "type"]].value_counts().reset_index()
    #     unique_combo_pruned = unique_combo.loc[unique_combo.Class != 'NO_SCORE_REGION']
    #     unique_all = ref[["Class"]].value_counts().reset_index()
    #     unique_all["type"] = "all"
    #     unique_all_pruned = unique_all.loc[unique_all.Class != 'NO_SCORE_REGION']
        
    #     combine_combo_pruned = pd.concat([unique_combo_pruned, unique_all_pruned])
    #     combine_combo_pruned.sort_values(["Class", "type"], inplace=True)

    #     final_combo_pruned = combine_combo_pruned.reset_index()
    #     final_combo_pruned = final_combo_pruned[["Class","type"]]

    #     if len(final_combo_pruned)>0:
    #         for i in range(len(final_combo_pruned)):
    #             y.append( [final_combo_pruned.loc[i, "Class"], final_combo_pruned.loc[i, "type"], 0.0, [0.0], [0.0], [0.0] ])
    #     else:
    #         logger.error("No matching Classes and types found in system output.")
    #         y.append( [ 'no_macthing_class', 'no_macthing_type', 0.0, [0.0, 0.0], [0.0, 1.0], [0.0, 0,0] ]) 
    # else:
    #     logger.error("No reference to score")
    #     y.append( ["NA", "NA", "NA", "NA", "NA", "NA"])
    # print("Y")
    # print(y)
    # return pd.DataFrame(y, columns=['Class', 'type', 'ap', 'precision', 'recall', 'llr'])


def segment_iou_v1(ref_start, ref_end, tgts):
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
    tIoU, intersection, union, collar_based_IoU : 1d array
        Temporal intersection over union score of the N's candidate segments and sub measurements.
    """
    #print("------------")
    #print(ref_start) 
    #print(ref_end)
    #print(tgts)
    ### normal IoU intersection
    sys_start, sys_end = [tgts[0], tgts[1]]
    tt1 = np.maximum(ref_start, sys_start)
    tt2 = np.minimum(ref_end, sys_end)    
    inter = (tt2 - tt1).clip(0)    # Segment intersection including Non-negative overlap score

    ### Collar based IoU has a different formula for the numerator
    ## MIN(MAX(re,se),MIN(re,se)+collar)  -   MAX(MIN(rs,s),MAX(rs,ss)-collar))
    collar = 15
    cb_tt1 = np.maximum(np.minimum(ref_start,tgts[0]),np.maximum(ref_start,tgts[0])-collar)
    cb_tt2 = np.minimum(np.maximum(ref_end  ,tgts[1]),np.minimum(ref_end,  tgts[1])+collar)
    cb_inter = (cb_tt2 - cb_tt1).clip(0)    # Segment intersection including Non-negative overlap score

    
    # Segment union.
    union = (sys_end - sys_start) + (ref_end - ref_start) - inter    
    tIoU = inter.astype(float) / union
    return tIoU, inter, union, cb_inter, cb_inter/union


def segment_iou_v2(sys_start, sys_end, refs, collar):
    """
    Compute __Temporal__ Intersection over Union (__IoU__) as defined in 
    [1], Eq.(1) given start and endpoints of intervals __g__ and __p__.    
    Vectorized impl. from ActivityNET/Pascal VOC devkit.

    Parameters
    ----------
    sys_start: float
        starting frame of hyp source segement
    sys_end: float
        end frame of source segement        
    refs : 2d array
        Temporal Ref test segments containing [starting x N, ending X N] times.
    collar : float
        The collar to use for scaled measures.  This is a SINGLE value because ONLY single files are present in refs

    Returns
    -------
    tIoU, intersection, union, shifted_sys_start, shifted_sys_end, scaled_pct_TP, scaled_pct_FP
        Temporal intersection over union score of the N's candidate segments and sub measurements.

    ### WARNING: shifted_sys_start, shifted_sys_end, scaled_pct_TP, scaled_pct_FP ARE simple lists ######
    """
    ### Unpack for computation
    ref_start, ref_end = [refs[0], refs[1]]
    
    #print("segment_iou_v2")
    #print(f"\nsys_start {sys_start} sys_end {sys_end} ----- ref_start {ref_start.to_list()} type {ref_end.to_list()}") 

    ### normal IoU intersection
    tt1 = np.maximum(ref_start, sys_start)
    tt2 = np.minimum(ref_end, sys_end)    
    inter = (tt2 - tt1).clip(0)    # Segment intersection including Non-negative overlap score

    ### Second variant - Collaring shifts the system towards the reference.
    ### csb = collared start 
    ###     = IF(rb<sb-15,sb-15,MIN(rb,sb))
    ### cse = collared end
    ###     = IF(re>se+15,se+15,MAX(re,se))
    ###         minENDs - maxBeg / (refDur)
    ### %TP = MIN(re,cse)- MAX(rb,csb))/(re-rb)
    ###       MIN(F33,Q33)-MAX(E33,P33))/(F33-E33)
    ###
    ###        Sys:   |XXXXXX------------YYYYY|
    ###        Ref:         |------------|
    ### For FP: %FP = 1
    ### For TP: %FP =  ( UncoveredRefBegin + UncoveredSysEnd )     /  SysDuration       
    ###                (IF(rb-csb>0,rb-csb,0)+IF(cse-re>0,cse-re,0))/(cse-csb)


    csb = [ sys_start-collar if (rs < sys_start - collar) else beg_min  for rs, beg_min in zip(ref_start, np.minimum(ref_start, sys_start)) ]
    #print(f"csb {csb}")
    
    cse = [ sys_end+collar if (re > sys_end + collar) else end_max for re, end_max in zip(ref_end, np.maximum(ref_end, sys_end)) ] 
    #print(f"cse {cse}")

    scaled_pct_TP = [ (np.minimum(re, ce)  - np.maximum(rs, cb)) / (re - rs) for rs, re, cb, ce in zip(ref_start, ref_end, csb, cse) ]
    #print(f"scaled_pct_TP {scaled_pct_TP}")

    scaled_pct_FP = [ ( (rs-cb if (rs-cb > 0) else 0) + (ce - re if (ce-re > 0) else 0)) / (re - rs) for rs, re, cb, ce in zip(ref_start, ref_end, csb, cse) ] 
    #print(f"scaled_pct_TP {scaled_pct_FP}")

    # Segment union.
    union = (sys_end - sys_start) + (ref_end - ref_start) - inter    
    tIoU = inter.astype(float) / union
    return tIoU, inter, union, csb, cse, scaled_pct_TP, scaled_pct_FP, collar


def compute_ious(row, ref, class_type, time_span_scale_collar, text_span_scale_collar):
    """
    Compute the ref/hyp matching table
    """
    refs = ref.loc[ ref['file_id'] == row.file_id ].copy() ### This filters by file ----  SUPER EXPENSIVE
    
    #print(f"\n\n------------------------------\nThe REF")
    #print(refs)
    #print(f"ROW - - - The HYP: file={row.file_id} start={row.start} end={row.end}")

    if len(refs) == 0:
        return pd.DataFrame(data=[[row.Class, None, row.type, row.file_id, np.nan, np.nan, row.start, row.end, row.llr, 0.0, row.status, None, None, None, row.start, row.end, 0.0, 1.0, None]],
            columns=['Class', 'Class_type', 'type', 'file_id', 'start_ref', 'end_ref', 'start_hyp', 'end_hyp', 'llr', 'IoU', 'hyp_status', 'length', 'intersection', 'union', 'shifted_sys_start', 'shifted_sys_end', 'pct_tp', 'pct_fp', 'scale_collar'])    
    
    else:        
        ### Set the scale collar based on the values (which are check for uniqueness) of type
        types = set(refs.type)
        assert len(types)==1, f"Internal Error: compute_ious() give a reference list with multiple source types: {types}"
        collar = text_span_scale_collar if (list(types)[0] == 'text') else time_span_scale_collar
        
        ### This computes the IoU regardless of the threshold for scoring.  We are going to 
        #refs['IoU'], refs['intersection'], refs['union'], refs['cb_intersection'], refs['cb_IoU'] = segment_iou_v1(row.start, row.end, [refs.start, refs.end])
        refs['IoU'], refs['intersection'], refs['union'], refs['shifted_sys_start'], refs['shifted_sys_end'], refs['pct_tp'],  refs['pct_fp'], refs['scale_collar'] = segment_iou_v2(row.start, row.end, [refs.start, refs.end], collar)  #####  ROW is the hyp #######
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

def compute_average_precision_tad(ref, hyp, Class, iou_thresholds, task, time_span_scale_collar, text_span_scale_collar):  
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
    iou_thresholds : a dictionary defining the thresholds.  e.g., {'IoU': 0.2}
        Temporal IoU Threshold (>=0)
    task:
        string that indicates task name. e.g. norm/emotion        
    time_span_scale_collar:
        The time span collar for scaled scoring
    text_span_scale_collar:
        The text span collar for scaled scoring

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
    #print(f"\n=========================================================================")
    #print(f"=============  compute_average_precision_tad Class={Class} =====================")
    #print(ref[((ref.Class == Class) | (ref.Class == 'NO_SCORE_REGION'))])
    #print(hyp[hyp.Class == Class])
    #print(f"=============  compute_average_precision_tad Class={Class} =====================")

    # REF has same amount of !score_regions for all runs, which need to be
    # excluded from overall REF count.
    npos = len(ref.loc[ref.Class.str.contains('NO_SCORE_REGION')==False])    
    output, out = {}, []
    alignment_df = pd.DataFrame()

    # No Class found.
    if hyp.empty:
        for iout in iou_thresholds:
            output[iout] = generate_zero_scores_norm_emotion(None)
        alignment_df = generate_all_fn_alignment_file(ref, task)
        return output,alignment_df

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', None)
    
    if (False):
        #print("------------Remove Noscores-----------")
        #print(ref.loc[ref.Class.str.contains('NO_SCORE_REGION') == True])
        #print("------------Residual Noscores-----------")
        ref = ref.loc[ref.Class.str.contains('NO_SCORE_REGION') == False]
        #print(ref)

    # Compute IoU for all hyps incl. NO_SCORE_REGION
    for idx, myhyp in hyp.iterrows():
        out.append(compute_ious(myhyp, ref, task, time_span_scale_collar, text_span_scale_collar))
    ihyp = pd.concat(out)
    #print("-----------------from compute_ious------")
    #print(ihyp)
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
            output[iout] = 0.0, [0.0], [0.0], [0.0]
        alignment_df = generate_all_fn_alignment_file(ref, task)
        return output,alignment_df

    #print("------------Adding FNs]--------------");
    ref_fn = ref.loc[~(ref["ref_uid"].isin(ihyp["ref_uid"])) & (ref.Class.str.contains('NO_SCORE_REGION') == False)]
    ref_fn = ref_fn.rename(columns={'start': 'start_ref', 'end': 'end_ref'})
    if (task == "norm"):
        ref_fn['hyp_status'] = "EMPTY_NA"
    ref_fn['pct_tp'] = 0.0
    ref_fn['pct_fp'] = 0.0
    ihyp = pd.concat([ihyp, ref_fn])
    #print(ihyp)
    #exit(0)
    
    # Sort by confidence score
    ihyp.sort_values(["llr"], ascending=False, inplace=True)        
    ihyp.reset_index(inplace=True, drop=True)

    #print("------------Pre AP Calc Alignment--------------");
    #print(ihyp)
    #exit(0)

    # Determine TP/FP @ IoU-Threshold
    for iout, params in iou_thresholds.items():
        ### This resets the tp and fp for each correctness threshold 
        ihyp[['tp', 'fp', 'md']] = [ 0, 1, 0 ] 
        ### Ref exists, above threshold (implying a TP)
        if (params['op'] == 'gte'):
            ihyp.loc[~ihyp['start_ref'].isna() & (ihyp[params['metric']] >= params['thresh']), ['tp', 'fp', 'md']] = [ 1, 0, 0 ] 
        if (params['op'] == 'gt'):
            ihyp.loc[~ihyp['start_ref'].isna() & (ihyp[params['metric']] > params['thresh']), ['tp', 'fp', 'md']] = [ 1, 0, 0 ] 
       
        # Mark TP as FP for duplicate ref matches at lower CS
        nhyp = ihyp.duplicated(subset = ['file_id', 'start_ref', 'end_ref', 'tp'], keep='first')
        ihyp.loc[ihyp.loc[nhyp == True].index, ['tp', 'fp', 'md']] = [ 0, 1, 0 ]
        ihyp.sort_values(["llr", "file_id"], ascending=[False, True], inplace=True)
        
        # Set MDs - This is because MD refs are added
        ihyp.loc [~ihyp['start_ref'].isna() & ihyp['start_hyp'].isna(), ['tp', 'fp', 'md']] = [ 0, 0, 1 ]

        ### Update the data for Class when IoU == 0.  This sets the class for No_score_regions        
        ihyp.loc[ihyp.loc[ihyp['Class'] == 'NO_SCORE_REGION'].index, ['start_ref', 'end_ref']] = [ float("nan"), float("nan") ]
        ihyp.loc[ihyp.loc[ihyp['IoU'] == 0.0].index, ['Class']] = [ Class ]
        
        # MDs are still in the alignment struct so the need to be removed for AP calc
        #print(f"--------Alignment for this threshold {params['metric']} >= {params['thresh']} --------------");
        #print(ihyp)
        #exit(0)
    
        ihyp["cum_tp"] = np.cumsum(ihyp.tp).astype(float)
        ihyp["cum_fp"] = np.cumsum(ihyp.fp).astype(float)

        ############ Beware: fhyp is the LAST row per LLR value to reduce the PR curve size!!!
        fhyp = ihyp
        thyp = fhyp.duplicated(subset = ['llr'], keep='last')
        fhyp = fhyp.loc[thyp == False]
        fhyp = fhyp.loc[fhyp.md == 0]   ### Remove the MDs before AP calc
        llr = np.array(fhyp["llr"])
        rec = (np.array(fhyp["cum_tp"]) / npos)
        prec = (np.array(fhyp["cum_tp"]) / (np.array(fhyp["cum_tp"]) + np.array(fhyp["cum_fp"])))

        ### OK, add some more metrics!
        nsys = fhyp.cum_tp.iat[-1] + fhyp.cum_fp.iat[-1] 
        scaled_recall =    ihyp.pct_tp.sum() / npos
        scaled_precision = ihyp.pct_tp.sum() / (ihyp.pct_tp.sum() + ihyp.pct_fp.sum())  
        measures = { 'AP': ap_interp(prec, rec), 
                     'prcurve:precision': prec,
                     'prcurve:recall': rec,
                     'prcurve:llr': llr,
                     'precision_at_MinLLR': prec[-1],
                     'recall_at_MinLLR': rec[-1],
                     'f1_at_MinLLR': f1(prec[-1], rec[-1]),
                     'llr_at_MinLLR': llr[-1],
                     'sum_tp_at_MinLLR': ihyp.tp.sum(),
                     'sum_fp_at_MinLLR': ihyp.fp.sum(),
                     'sum_scaled_tp_at_MinLLR': ihyp.pct_tp.sum(),
                     'sum_scaled_fp_at_MinLLR': ihyp.pct_fp.sum(),
                     'scaled_recall_at_MinLLR':  scaled_recall,
                     'scaled_precision_at_MinLLR':  scaled_precision,
                     'scaled_f1_at_MinLLR':  f1(scaled_precision, scaled_recall),
                    }
       
        output[iout] = measures
 
        ihyp_fields = ["Class","type","tp","fp","file_id","start_ref","end_ref","start_hyp","end_hyp","IoU","llr","intersection", "union", 'shifted_sys_start', 'shifted_sys_end', 'pct_tp', 'pct_fp', 'scale_collar']
        if (task == "norm"):
            ihyp_fields.append("status")
            ihyp_fields.append("hyp_status")
        ihyp = ihyp[ihyp_fields]
        
        alignment_df = pd.concat([alignment_df, ihyp])
        #print(f"-------------------------Alignment_df for {iout}--------------");
        #print(ihyp)
        #for key, value in output[iout].items():
        #    print(f"   {key} -> {value}")
        #exit(0)
        
    final_alignment_df = generate_alignment_file(ref.loc[ref.Class.str.contains('NO_SCORE_REGION')==False], alignment_df, task)
    #print("late2")
    #print("Alignment")
    #print(alignment_df)
    #print("Alignment Output")
    #print(final_alignment_df)
    #exit(0)

    return output,final_alignment_df

def compute_multiclass_iou_pr(ref, hyp, iou_thresholds, mapping_df, class_type, time_span_scale_collar, text_span_scale_collar):
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
    iou_thresholds: a dictionary of thresholds.  defines the metric, operation, and value
        List of IoU levels to score at.
    mapping_df:
        norm mapping dataframe
    class_type:
        string that indicates task name. e.g. norm/emotion
    time_span_scale_collar:
        The time span collar for scaled scoring
    text_span_scale_collar:
        The text span collar for scaled scoring


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
    pr_scores = {}
    [ pr_scores.setdefault(iout, []) for iout in iou_thresholds ]
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

        ### Filter the DF by the class and type to compute AP.  This means the both collars need passed so that that code can make decision on which to use.
        ### apScore is dict (for IoU) and a dict (for measures)
        apScore, alignment = compute_average_precision_tad(
                ref=ref.loc[((ref.Class == final_combo_pruned.loc[i, "Class"]) | (ref.Class == 'NO_SCORE_REGION')) & (ref.type.isin(match_type))].reset_index(drop=True),                        
                hyp=hyp_scoring,
                Class=final_combo_pruned.loc[i, "Class"],
                iou_thresholds=iou_thresholds,
                task=class_type, 
                time_span_scale_collar=time_span_scale_collar,
                text_span_scale_collar=text_span_scale_collar)
        
        #apScores.append(apScore)
        if final_combo_pruned.loc[i, "type"] == "all":
            alignment_df = pd.concat([alignment_df, alignment])

        #print("-----------------------")
        #print(apScore)
        ### Load the results into the IoU-specific data frame.  The first time the IoU is found, the DF is began
        for iout in iou_thresholds:
            ### Add Values then append
            #print(f"iout {iout}")
            #print(apScore[iout])
            #print(final_combo_pruned.loc[i])
            apScore[iout]['Class'] = final_combo_pruned.loc[i, "Class"]
            apScore[iout]['type'] = final_combo_pruned.loc[i, "type"]
            pr_scores[iout].append(apScore[iout])

    final_alignment_df = alignment_df.drop_duplicates() ### Good heavens, this must take a TON of time. IT's needed if multiple IoU thresholds are used

    # print("SCORING COMPLETE")
    # for iout, val in pr_scores.items():
    #     print(iout)
    #     for sc in range(len(pr_scores[iout])):
    #         for met, met_val in pr_scores[iout][sc].items():
    #             print(f"   {sc} {met} -> {met_val}")
    # print(final_alignment_df)
        
    # exit(0)
    # exit(0)
    
    # for i in range(len(final_combo_pruned)):
    #     for iout in iou_thresholds:
    #         #print(f"i{i} iout{iout}")
    #         #print(apScores[i][iout])
    #         scores[iout].append([final_combo_pruned.loc[i, "Class"], final_combo_pruned.loc[i, "type"], apScores[i][iout][0], apScores[i][iout][1], apScores[i][iout][2], apScores[i][iout][3]])

    # # Build results.  This is a dict of dataframes with the key being IoU.  SO it is pooled by IoU.  
    # pr_scores = {}
    # for iout in iou_thresholds: 
    #     pr_scores[iout] = pd.DataFrame(scores[iout], columns = ['Class', 'type', 'ap', 'precision', 'recall', 'llr'])
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
        
        map_scores["correctness_criteria"] = "{%s}" % iout
        map_scores["metric"] = "mAP"

        if class_type == "norm":
            map_scores["task"] = "nd"
        if class_type == "emotion":
            map_scores["task"] = "ed"
        
        map_scores = map_scores[["task","genre","metric","value","correctness_criteria"]]
        map_scores_threshold = pd.concat([map_scores, map_scores_threshold])
    
    map_scores_threshold.to_csv(os.path.join(output_dir, "scores_aggregated.tab"), sep = "\t", index = None)
    
# def sumup_tad_class_level_scores_orig(pr_iou_scores, iou_thresholds, output_dir):
#     """
#     Write class level result into a file
#     """
#     prs_threshold = pd.DataFrame()   
#     for iout in sorted(iou_thresholds):        
#         prs = pr_iou_scores[iout]
#         if prs["ap"].values[0] != "NA":
#             prs.ap = prs.ap.round(3)
#         prs["metric"] = "AP"        
#         prs["correctness_criteria"] = "{%s}" % iout
#         prs = prs.rename(columns={'Class': 'class', 'type': 'genre', 'ap': 'value'})
#         prs = prs[["class","genre","metric","value","correctness_criteria"]]
#         prs_threshold = pd.concat([prs, prs_threshold])
    
#     prs_threshold.to_csv(os.path.join(output_dir, "scores_by_class.tab"), sep = "\t", index = None)


        
def sumup_tad_class_level_scores(pr_iou_scores, iou_thresholds, output_dir, class_type):
    """
    Write class level result into a file
    """
    #print(">>sumup_tad_class_level_scores")

    ### Build the class table
    table = []   #class   genre   metric  value   correctness_criteria
    for iout in sorted(iou_thresholds):
        prs = pr_iou_scores[iout]  ### This is an array of class, type, * scores
        for row in range(len(prs)):
            Class = prs[row]['Class']
            Type = prs[row]['type']

            metrics = list(prs[row].keys())
            metrics.sort()
            for metric in metrics:
                if (metric not in ['prcurve:precision', 'prcurve:recall', 'prcurve:llr', 'Class', 'type' ]):
                    table.append([ Class, Type, metric, np.round(prs[row][metric], 3) if (prs[row][metric] is not None) else prs[row][metric], "{%s}" % iout] )
                    if (metric == 'AP'):  ### Add twice as a standard name
                        table.append([ Class, Type, 'average_precision', np.round(prs[row][metric], 3) if (prs[row][metric] is not None) else prs[row][metric], "{%s}" % iout] )


            ### Add the PR curve
            if (prs[row]['prcurve:precision'] is not None):
                d = { "precision": [ x for x in prs[row]['prcurve:precision'] ],
                      "recall": [ x for x in prs[row]['prcurve:recall'] ],
                      "llr": [ x for x in prs[row]['prcurve:llr'] ]
                     }
                table.append([ Class, Type, "PRCurve_json", json.dumps(d), "{%s}" % iout] )

    table_df = pd.DataFrame(table, columns=["class", "genre", "metric", "value", "correctness_criteria"])
    table_df.to_csv(os.path.join(output_dir, "scores_by_class.tab"), sep = "\t", index = None)
    #print(table_df)

    ### Build the aggregated table from table_df
    agg_table = table_df[table_df.metric != 'PRCurve_json'].groupby(['genre', "metric", "correctness_criteria"])['value'].mean().reset_index()
    agg_table['metric'] = 'mean_' + agg_table['metric']
    agg_table.loc[agg_table.metric == 'mean_AP', ['metric']] = [ 'mAP' ]      ## Rename AP to mAP
    agg_table['task'] = 'nd' if (class_type == 'norm') else 'ed'
    agg_table[["task", "genre", "metric", "value", "correctness_criteria"]].to_csv(os.path.join(output_dir, "scores_aggregated.tab"), sep = "\t", index = None)

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

        ### Make the long form!!!
        ### AP
        temp = results[cp].copy(deep=True)
        temp['metric'] = "AP"
        temp = temp[["class","genre","metric","value","correctness_criteria"]]
        results_long = pd.concat([results_long, temp]);
        ### PRCurve
        temp = results[cp].copy(deep=True)
        temp['metric'] = "PRCurve_json"
        d = {"precision": temp['precision'].tolist()[0], 'recall': temp['recall'].tolist()[0], 'llr': temp['llr'].tolist()[0]}
        temp['value'] = json.dumps(d)
        
        temp = temp[["class","genre","metric","value","correctness_criteria"]]
        results_long = pd.concat([results_long, temp]);

    results_long.to_csv(os.path.join(output_dir, "scores_by_class.tab"), sep = "\t", index = None, quoting=None)   

def score_tad(ref, hyp, class_type, iou_thresholds, output_dir, mapping_df, time_span_scale_collar, text_span_scale_collar):
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
    iou_thresholds: a dictionary of thresholds.  defines the metric, operation, and value
        List of IoU levels to score at.
    output_dir: str
        Path to a directory (created on demand) for output files   
    mapping_df:
        norm mapping dataframe 
    time_span_scale_collar:
        The time span collar for scaled scoring
    text_span_scale_collar:
        The text span collar for scaled scoring
    """    
    # FIXME: Use a No score-region parameter
    tad_add_noscore_region(ref,hyp)
    #print(">> Enter score_tad()")
    #print("REF - With Noscore Regions")
    #print(ref)
    #print("HYP")
    #print(hyp)
    if len(ref) > 0:
        if len(hyp) > 0:
            pr_iou_scores, final_alignment_df = compute_multiclass_iou_pr(ref, hyp, iou_thresholds, mapping_df, class_type, time_span_scale_collar, text_span_scale_collar)
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
    graph_info_dict = []
    generate_alignment_statistics(final_alignment_df, class_type, output_dir, info_dict = graph_info_dict)

    sumup_tad_class_level_scores(pr_iou_scores, iou_thresholds, output_dir, class_type)
    grqph_info_dict = make_pr_curve(pr_iou_scores, class_type, class_type, output_dir = output_dir, info_dict = graph_info_dict)
    graph_info_df = pd.DataFrame(graph_info_dict)
    graph_info_df.to_csv(os.path.join(output_dir, "graph_info.tab"), index = False, quoting=3, sep="\t", escapechar="\t")
    
