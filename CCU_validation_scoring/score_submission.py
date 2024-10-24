import os
import pandas as pd
import logging
import re
from fractions import Fraction
from .preprocess_reference import *
from .utils import *
from .build_statistic import *
from .score_norm_emotion import score_tad
from .score_valence_arousal import *
from .score_changepoint import *

logger = logging.getLogger('SCORING')

def parse_thresholds(arg):
    ''' Returns the parsed, normalized thresholds
    '''
    
    dic = {}
    succeed = True
    for item in arg.split(','):
        match = re.match('^([\\d]*\\.[\\d]+|[\\d]+|[\\d]+\\.)$', item)
        if (match is not None):
            dic['iou=' + match.group()] = {'metric': 'IoU', 'op': 'gte', 'thresh': float(match.group())}
        else:
            match = re.match('^(iou|intersection):(gt|gte):([\\d]*\\.[\\d]+|[\\d]+|[\\d]+\\.)$', item)
            if (match is not None):
                met = match.group(1)
                if met == "iou":
                        met = "IoU"
                dic[item] = { 'metric':met, 'op': match.group(2), 'thresh': float(match.group(3)) }
            else:
                succeed = False
                print(f"Error: Could not parse the threshold {item}.  Options are comma-separated variants of:\n    #\n    (iou|intersection):(gt|gte):#\n")
                
    assert succeed, "Parse thresholds failed"
    return(dic)

def parse_llr_filter(arg):
    ''' Returns the parsed llr_filter
    '''
    
    dic = { 'filter_order': None, 'filter_def': None, 'filter_threshold': None}
    succeed = True
    if (arg != ""):
        match = re.match('^(after_read|after_transforms):(by_value):([\\d]*\\.[\\d]+|[\\d]+|[\\d]+\\.)$', arg)
        if (match is not None):
            dic['filter_order'] = match.group(1)
            dic['filter_method'] = match.group(2)
            dic['filter_threshold'] = match.group(3)
        else:
            assert False, f"Parse thresholds failed /{arg}/"
    return(dic)

def get_contained_index(ref_row, hyp):
        if (ref_row['Class'] != "noann"):
                return([])
        return hyp[(ref_row['file_id'] == hyp['file_id']) & (ref_row['start'] <= hyp['start']) & (ref_row['end'] >= hyp['end'])].index

def pre_filter_system_in_noann_region(hyp, ref):
		"""
		Remove any system instances WHOLLY Contained intial or final file noscore regions
		"""
		hyp.reset_index()
		for fileid in set(ref['file_id']):
				sref = ref[(ref['file_id'] == fileid) & (ref.Class == 'noann') ]
				#print(sref)
				### Straight drop of instances within the begin/end of *ALL* NoScore Regions
				for ind in sref.index:
					hyp.drop(get_contained_index(sref.loc[ind], hyp),  inplace = True)

				### Reset system boundaries to the begin of the NoScore
				if (True):
					for ind in sref.index:  ### These are ONLY noscore Refs
						for index, row in hyp[hyp.file_id == fileid].iterrows():
							Type = list(sref["type"])[0]
							#print(f"index hyp[{index}] {row.Class} H:{row.start}:{row.end} ?? R:{sref.loc[ind].start}:{sref.loc[ind].end}")
							if (row.start < sref.loc[ind].start and sref.loc[ind].end < row.end): ### Full overlap
								#print("Shucks")
								hyp.at[index,'hyp_uid'] = row.hyp_uid + "-UnDroppedFullOverlap"
								#x=1
							else:
								if (row.start < sref.loc[ind].end and sref.loc[ind].end < row.end):  ### Straddles the boundary
									if Type == "text":
										hyp.at[index,'start'] = sref.loc[ind].end + 1
									else:
										hyp.at[index,'start'] = sref.loc[ind].end
									hyp.at[index,'hyp_uid'] = row.hyp_uid + "-TruncStart"
									hyp.at[index,'hyp_isTruncated'] = True
									#print(f"   Truncated start H:{hyp.at[index,'start']}:{hyp.at[index,'end']}")
									#x=1
								if (row.start < sref.loc[ind].start and sref.loc[ind].start < row.end):  ### Straddles the boundary
									if Type == "text":
										hyp.at[index,'end'] = sref.loc[ind].start - 1
									else:
										hyp.at[index,'end'] = sref.loc[ind].start
									hyp.at[index,'hyp_uid'] = row.hyp_uid + "-TruncEnd"
									hyp.at[index,'hyp_isTruncated'] = True
									#x=1
							#print(f"   Truncated end H:{hyp.at[index,'start']}:{hyp.at[index,'end']}")
				
		#print(hyp[hyp.Class == '108'])
		#exit(0)
		return(hyp)

def set_text_gap(arg):
    if (arg):
        return(int(arg))
    else:
        return(None)

def set_time_gap(arg):
    if (arg):
        return(float(arg))
    else:
        return(None)
    
def score_nd_submission_dir_cli(args):

	try:
		scoring_index = pd.read_csv(args.scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:SCORING:{} is not a valid scoring index file'.format(args.scoring_index_file))
		exit(1)

	llr_filter = parse_llr_filter(args.llr_filter)

	check_scoring_index_out_of_scope(args.reference_dir, scoring_index, "norms")

	merge_ref_text_gap = set_text_gap(args.merge_ref_text_gap)
	merge_ref_time_gap = set_time_gap(args.merge_ref_time_gap)

	ensure_output_dir(args.output_dir)
	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "norms", text_gap = merge_ref_text_gap, time_gap = merge_ref_time_gap, merge_label = args.merge_ref_label, dump_inputs=args.dump_inputs, output_dir=args.output_dir, fix_ref_status_conflict_label=args.fix_ref_status_conflict, minimum_vote_agreement=None)

	if args.norm_list_file:
		ref = process_subset_norm_emotion(args.norm_list_file, ref)
	#print(f"post processesd Ref merge_label={args.merge_ref_label}")
	#print(ref)
	hyp = preprocess_submission_file(args.submission_dir, args.reference_dir, scoring_index, "norms", args.submission_format)
	if (args.dump_inputs):
		hyp.to_csv(os.path.join(args.output_dir, "inputs.sys.read.tab"), sep = "\t", index = None)

	if (llr_filter['filter_order'] is not None and llr_filter['filter_order'] == 'after_read'):
		## only by_value :)
		hyp = hyp.drop(hyp[ hyp['llr'] < float(llr_filter['filter_threshold']) ].index)
                
	if args.mapping_submission_dir:
		mapping_file = os.path.join(args.mapping_submission_dir, "nd.map.tab")
		mapping_df = pd.read_csv(mapping_file, dtype="object", sep = "\t")
	else:
		mapping_df = None

	if args.merge_sys_text_gap:
		merge_sys_text_gap = int(args.merge_sys_text_gap)
	else:
		merge_sys_text_gap = None

	if args.merge_sys_time_gap:
		merge_sys_time_gap = float(args.merge_sys_time_gap)
	else:
		merge_sys_time_gap = None

	if hyp.shape[0] > 0:
		hyp = pre_filter_system_in_noann_region(hyp, ref)
		
	merged_hyp = merge_sys_instance(hyp, merge_sys_text_gap, merge_sys_time_gap, args.combine_sys_llrs, args.merge_sys_label, "norms")

	thresholds = parse_thresholds(args.iou_thresholds)

	if ref.shape[0] > 0:
		ref = ref[ref.Class != "none"]
		ref['ref_uid'] = [ "R"+str(s) for s in range(len(ref['file_id'])) ] ### This is a unique REF ID to help find FN
		ref['isNSCR'] = ref.Class.isin(['noann', 'NO_SCORE_REGION'])

	statistic(args.reference_dir, ref, args.submission_dir, merged_hyp, args.output_dir, "norms")

	if (llr_filter['filter_order'] is not None and llr_filter['filter_order'] == 'after_transforms'):
		merged_hyp = merged_hyp.drop(merged_hyp[ merged_hyp['llr'] < float(llr_filter['filter_threshold']) ].index)

	if (args.dump_inputs):
		ref.to_csv(os.path.join(args.output_dir, "inputs.ref.scored.tab"), sep = "\t", index = None)
		merged_hyp.to_csv(os.path.join(args.output_dir, "inputs.sys.scored.tab"), sep = "\t", index = None)
	score_tad(ref, merged_hyp, "norm", thresholds, args.output_dir, mapping_df, float(args.time_span_scale_collar), float(args.text_span_scale_collar), args.align_hacks)
	generate_scoring_parameter_file(args)

	if (not args.quiet):
		print("Alignment")
		print("---------------")
		print(open(os.path.join(args.output_dir, 'instance_alignment.tab')).read())
		print("Class Scores")
		print("---------------")
		print(open(os.path.join(args.output_dir, 'scores_by_class.tab')).read())
		print("Aggregated Scores")
		print("-------------")
		print(open(os.path.join(args.output_dir, 'scores_aggregated.tab')).read())
    
def score_ed_submission_dir_cli(args):

	try:
		scoring_index = pd.read_csv(args.scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:SCORING:{} is not a valid scoring index file'.format(args.scoring_index_file))
		exit(1)

	llr_filter = parse_llr_filter(args.llr_filter)

	check_scoring_index_out_of_scope(args.reference_dir, scoring_index, "emotions")
	
	merge_ref_text_gap = set_text_gap(args.merge_ref_text_gap)
	merge_ref_time_gap = set_time_gap(args.merge_ref_time_gap)

	ensure_output_dir(args.output_dir)
	if args.file_merge_proportion:
		file_merge_proportion = float(Fraction(args.file_merge_proportion))
		ref,seg = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "emotions", text_gap = merge_ref_text_gap, time_gap = merge_ref_time_gap, dump_inputs=args.dump_inputs, output_dir=args.output_dir, minimum_vote_agreement=args.minimum_vote_agreement, file_merge_proportion=file_merge_proportion)
	else:
		ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "emotions", text_gap = merge_ref_text_gap, time_gap = merge_ref_time_gap, dump_inputs=args.dump_inputs, output_dir=args.output_dir, minimum_vote_agreement=args.minimum_vote_agreement)
	if args.emotion_list_file:
		ref = process_subset_norm_emotion(args.emotion_list_file, ref)
	hyp = preprocess_submission_file(args.submission_dir, args.reference_dir, scoring_index, "emotions")
	if (args.dump_inputs):
		hyp.to_csv(os.path.join(args.output_dir, "inputs.sys.read.tab"), sep = "\t", index = None)

	if (llr_filter['filter_order'] is not None and llr_filter['filter_order'] == 'after_read'):
		## only by_value :)
		hyp = hyp.drop(hyp[ hyp['llr'] < float(llr_filter['filter_threshold']) ].index)

	if args.merge_sys_text_gap:
		merge_sys_text_gap = int(args.merge_sys_text_gap)
	else:
		merge_sys_text_gap = None

	if args.merge_sys_time_gap:
		merge_sys_time_gap = float(args.merge_sys_time_gap)
	else:
		merge_sys_time_gap = None

	if args.file_merge_proportion:
		file_merge_proportion = float(Fraction(args.file_merge_proportion))
		final_hyp = file_based_merge_sys(hyp, seg)
	else:
		if hyp.shape[0] > 0:
			hyp = pre_filter_system_in_noann_region(hyp, ref)
		final_hyp = merge_sys_instance(hyp, merge_sys_text_gap, merge_sys_time_gap, args.combine_sys_llrs, args.merge_sys_label, "emotions")

	thresholds = parse_thresholds(args.iou_thresholds)

	if ref.shape[0] > 0:
		ref = ref[ref.Class != "none"]
		ref['ref_uid'] = [ "R"+str(s) for s in range(len(ref['file_id'])) ] ### This is a unique REF ID to help find FN
		ref['isNSCR'] = ref.Class.isin(['noann', 'NO_SCORE_REGION'])

	if (llr_filter['filter_order'] is not None and llr_filter['filter_order'] == 'after_transforms'):
		final_hyp = final_hyp.drop(final_hyp[ final_hyp['llr'] < float(llr_filter['filter_threshold']) ].index)

	statistic(args.reference_dir, ref, args.submission_dir, final_hyp, args.output_dir, "emotions")
	if (args.dump_inputs):
		ref.to_csv(os.path.join(args.output_dir, "inputs.ref.scored.tab"), sep = "\t", index = None)
		final_hyp.to_csv(os.path.join(args.output_dir, "inputs.sys.scored.tab"), sep = "\t", index = None)

	score_tad(ref, final_hyp, "emotion", thresholds, args.output_dir, None, float(args.time_span_scale_collar), float(args.text_span_scale_collar), args.align_hacks)
	generate_scoring_parameter_file(args)
        
	if (not args.quiet):
		print("Alignment")
		print("---------------")
		print(open(os.path.join(args.output_dir, 'instance_alignment.tab')).read())
		print("Class Scores")
		print("---------------")
		print(open(os.path.join(args.output_dir, 'scores_by_class.tab')).read())
		print("Aggregated Scores")
		print("-------------")
		print(open(os.path.join(args.output_dir, 'scores_aggregated.tab')).read())            

def score_vd_submission_dir_cli(args):

	try:
		scoring_index = pd.read_csv(args.scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:SCORING:{} is not a valid scoring index file'.format(args.scoring_index_file))
		exit(1)

	check_scoring_index_out_of_scope(args.reference_dir, scoring_index, "valence_continuous")
	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "valence_continuous")
	hyp = preprocess_submission_file(args.submission_dir, args.reference_dir, scoring_index, "valence_continuous", gap_allowed = args.gap_allowed)

	if ref.shape[0] > 0:
		ref['ref_uid'] = [ "R"+str(s) for s in range(len(ref['file_id'])) ] ### This is a unique REF ID to help find FN
		ref['isNSCR'] = ref.Class.isin(['noann', 'NO_SCORE_REGION'])

	ensure_output_dir(args.output_dir)
	statistic(args.reference_dir, ref, args.submission_dir, hyp, args.output_dir, "valence_continuous")
	score_valence_arousal(ref, hyp, output_dir = args.output_dir, task = "valence_continuous")
	generate_scoring_parameter_file(args)	
	
	if (not args.quiet):
		print("Diarization")
		print("---------------")
		print(open(os.path.join(args.output_dir, 'segment_diarization.tab')).read())
		print("Aggregated Scores")
		print("-------------")
		print(open(os.path.join(args.output_dir, 'scores_aggregated.tab')).read())
	

def score_ad_submission_dir_cli(args):

	try:
		scoring_index = pd.read_csv(args.scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:SCORING:{} is not a valid scoring index file'.format(args.scoring_index_file))
		exit(1)

	check_scoring_index_out_of_scope(args.reference_dir, scoring_index, "arousal_continuous")
	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "arousal_continuous")
	hyp = preprocess_submission_file(args.submission_dir, args.reference_dir, scoring_index, "arousal_continuous", gap_allowed = args.gap_allowed)

	if ref.shape[0] > 0:
		ref['ref_uid'] = [ "R"+str(s) for s in range(len(ref['file_id'])) ] ### This is a unique REF ID to help find FN
		ref['isNSCR'] = ref.Class.isin(['noann', 'NO_SCORE_REGION'])

	ensure_output_dir(args.output_dir)
	statistic(args.reference_dir, ref, args.submission_dir, hyp, args.output_dir, "arousal_continuous")

	score_valence_arousal(ref, hyp, output_dir = args.output_dir, task = "arousal_continuous")
	generate_scoring_parameter_file(args)	
	
	if (not args.quiet):
		print("Diarization")
		print("---------------")
		print(open(os.path.join(args.output_dir, 'segment_diarization.tab')).read())
		print("Aggregated Scores")
		print("-------------")
		print(open(os.path.join(args.output_dir, 'scores_aggregated.tab')).read())

def score_cd_submission_dir_cli(args):

	llr_filter = parse_llr_filter(args.llr_filter)

	try:
		scoring_index = pd.read_csv(args.scoring_index_file, usecols = ['file_id'], sep = "\t")
	except Exception as e:
		logger.error('ERROR:SCORING:{} is not a valid scoring index file'.format(args.scoring_index_file))
		exit(1)

	check_scoring_index_out_of_scope(args.reference_dir, scoring_index, "changepoint")
	ref = preprocess_reference_dir(ref_dir = args.reference_dir, scoring_index = scoring_index, task = "changepoint")
	hyp = preprocess_submission_file(args.submission_dir, args.reference_dir, scoring_index, "changepoint")

	if ref.shape[0] > 0:
		ref['ref_uid'] = [ "R"+str(s) for s in range(len(ref['file_id'])) ] ### This is a unique REF ID to help find FN
		ref['isNSCR'] = ref.Class.isin(['noann', 'NO_SCORE_REGION'])

	if (args.dump_inputs):
		hyp.to_csv(os.path.join(args.output_dir, "inputs.sys.read.tab"), sep = "\t", index = None)

	if (llr_filter['filter_order'] is not None and llr_filter['filter_order'] == 'after_read'):
		## only by_value :)
		hyp = hyp.drop(hyp[ hyp['llr'] < float(llr_filter['filter_threshold']) ].index)

	text_thresholds = [int(i) for i in args.delta_cp_text_thresholds.split(',')]
	time_thresholds = [float(i) for i in args.delta_cp_time_thresholds.split(',')]

	if (llr_filter['filter_order'] is not None and llr_filter['filter_order'] == 'after_transforms'):
		hyp = hyp.drop(hyp[ hyp['llr'] < float(llr_filter['filter_threshold']) ].index)
        
	ensure_output_dir(args.output_dir)
	statistic(args.reference_dir, ref, args.submission_dir, hyp, args.output_dir, "changepoint")
	if (args.dump_inputs):
		ref.to_csv(os.path.join(args.output_dir, "inputs.ref.scored.tab"), sep = "\t", index = None)
		hyp.to_csv(os.path.join(args.output_dir, "inputs.sys.scored.tab"), sep = "\t", index = None)

	score_cp(ref, hyp, delta_cp_text_thresholds=text_thresholds, delta_cp_time_thresholds=time_thresholds, output_dir=args.output_dir)
	generate_scoring_parameter_file(args)
	
	if (not args.quiet):
		print("Alignment")
		print("---------------")
		print(open(os.path.join(args.output_dir, 'instance_alignment.tab')).read())
		print("Class Scores")
		print("---------------")
		print(open(os.path.join(args.output_dir, 'scores_by_class.tab')).read())







