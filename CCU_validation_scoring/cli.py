import argparse
from . import validate_submission
from . import validate_reference
from . import score_submission

def print_package_version(args):
    print(__import__(__package__).__version__)

def check_valid_argument_pair_merge_sys(args):

    if "score_nd" in str(vars(args)['func']):
        if not args.merge_sys_text_gap and not args.merge_sys_time_gap and not args.combine_sys_llrs and not args.merge_sys_label:
            return True
        elif args.merge_sys_text_gap and args.merge_sys_time_gap and args.combine_sys_llrs and args.merge_sys_label:
            return True
        elif args.merge_sys_text_gap and not args.merge_sys_time_gap and args.combine_sys_llrs and args.merge_sys_label:
            return True
        elif not args.merge_sys_text_gap and args.merge_sys_time_gap and args.combine_sys_llrs and args.merge_sys_label:
            return True
        else:
            return False

    elif "score_ed" in str(vars(args)['func']):
        if not args.merge_sys_text_gap and not args.merge_sys_time_gap and not args.combine_sys_llrs and not args.merge_sys_label:
            return True
        elif args.merge_sys_text_gap and args.merge_sys_time_gap and args.combine_sys_llrs and args.merge_sys_label:
            return True
        elif args.merge_sys_text_gap and not args.merge_sys_time_gap and args.combine_sys_llrs and args.merge_sys_label:
            return True
        elif not args.merge_sys_text_gap and args.merge_sys_time_gap and args.combine_sys_llrs and args.merge_sys_label:
            return True
        else:
            return False

    else:
        return True
    
def check_valid_argument_pair_merge_ref(args):

    if "score_nd" in str(vars(args)['func']):
        if not args.merge_ref_text_gap and not args.merge_ref_time_gap and not args.merge_ref_label:
            return True
        elif args.merge_ref_text_gap and args.merge_ref_time_gap and args.merge_ref_label:
            return True
        elif args.merge_ref_text_gap and not args.merge_ref_time_gap and args.merge_ref_label:
            return True
        elif not args.merge_ref_text_gap and args.merge_ref_time_gap and args.merge_ref_label:
            return True
        else:
            return False

    else:
        return True

def main():
        
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(title='subcommands')

    # Add version subcommand
    version_parser = subs.add_parser('version', description='Print package version')
    version_parser.set_defaults(func=print_package_version)

    validate_ref_parser = subs.add_parser('validate-ref', description='Validate the reference directory')
    validate_ref_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')

    validate_ref_parser.set_defaults(func=validate_reference.validate_ref_submission_dir_cli)
   
    # ADD ANOTHER SUBCOMMAND FOR INDEX FILE
    
    validate_nd_parser = subs.add_parser('validate-nd', description='Validate a norm discovery submission directory')
    validate_nd_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing a norm detection submission')
    validate_nd_parser.add_argument('-sf','--submission-format', type=str, default="regular", choices=['regular','open'], help='Choose submission format, regular is for CCU evaluation while open is for OpenCCU')
    validate_nd_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
    validate_nd_parser.add_argument('-n', '--norm_list_file', type=str, required=False, help="Use to validate norm string during evaluation")

    validate_nd_parser.set_defaults(func=validate_submission.validate_nd_submission_dir_cli)

    validate_ed_parser = subs.add_parser('validate-ed', description='Validate a emotion detection submission directory')
    validate_ed_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing a emotion detection submission')
    validate_ed_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')

    validate_ed_parser.set_defaults(func=validate_submission.validate_ed_submission_dir_cli)

    validate_vd_parser = subs.add_parser('validate-vd', description='Validate a valence diarization submission directory')
    validate_vd_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing a valence diarization submission')
    validate_vd_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
    validate_vd_parser.add_argument("-g", "--gap-allowed", action='store_true', default=False, help="Allow gap in system output of valence and arousal")

    validate_vd_parser.set_defaults(func=validate_submission.validate_vd_submission_dir_cli)

    validate_ad_parser = subs.add_parser('validate-ad', description='Validate an arousal diarization submission directory')
    validate_ad_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing an arousal diarization submission')
    validate_ad_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
    validate_ad_parser.add_argument("-g", "--gap-allowed", action='store_true', default=False, help="Allow gap in system output of valence and arousal")

    validate_ad_parser.set_defaults(func=validate_submission.validate_ad_submission_dir_cli)

    validate_cd_parser = subs.add_parser('validate-cd', description='Validate a change detection submission directory')
    validate_cd_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing a change detection submission')
    validate_cd_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')

    validate_cd_parser.set_defaults(func=validate_submission.validate_cd_submission_dir_cli)

    validate_ndmap_parser = subs.add_parser('validate-ndmap', description='Validate a norm discovery mapping submission directory')
    validate_ndmap_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing a norm mapping submission')
    validate_ndmap_parser.add_argument('-n', '--hidden-norm-list-file', type=str, required=True, help="Use to validate ref_norm in mapping file")

    validate_ndmap_parser.set_defaults(func=validate_submission.validate_ndmap_submission_dir_cli)    

    score_nd_parser = subs.add_parser('score-nd', description='Score a norm discovery submission directory')
    score_nd_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing a norm detection submission')
    score_nd_parser.add_argument('-sf','--submission-format', type=str, default="regular", choices=['regular','open'], help='Choose submission format, regular is for CCU evaluation while open is for OpenCCU')
    score_nd_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
    score_nd_parser.add_argument('-i','--scoring-index-file', type=str, required=True, help='Use to filter file from scoring (REF)')
    score_nd_parser.add_argument('-m','--mapping-submission-dir', type=str, help='Directory containing a norm mapping submission')
    score_nd_parser.add_argument('-n', '--norm_list_file', type=str, required=False, help="Use to filter norm from scoring (REF)")
    score_nd_parser.add_argument("-t", "--iou_thresholds", nargs='?', default="0.2", help="A comma separated list of IoU thresholds and the default value is 0.2.  Alternative criteria can be used by specifying the metric and its value using the form <METRIC>:<OPERATION>:<VALUE>.  'iou:gt:0.2' is the default value. The defined <METRIC>s are 'iou' and 'intersection'. The operations are 'gt' for '>', or 'gte' for '>='")
    score_nd_parser.add_argument("-aC", "--time_span_scale_collar", nargs='?', default="15", help="Forgiveness collar (in seconds) to determine true positive or false positive in scale metric alignment")
    score_nd_parser.add_argument("-xC", "--text_span_scale_collar", nargs='?', default="150", help="Forgiveness collar (in characters) to determine true positive or false positive in scale metric alignment")
    score_nd_parser.add_argument("-o", "--output_dir", type=str, nargs='?', default="tmp", help="Output directory")
    score_nd_parser.add_argument("-xR", "--merge_ref_text_gap", type=str, required=False, help="Merge reference text gap character")
    score_nd_parser.add_argument("-aR", "--merge_ref_time_gap", type=str, required=False, help="Merge reference time gap second")
    score_nd_parser.add_argument("-vR", "--merge_ref_label", type=str, choices=['class', 'class-status'], required=False, help="Choose class or class-status to define how to handle the adhere/violate labels for the reference norm instances merging. class is to use the class label only (ignoring status) to merge and class-status is to use the class and status label to merge")
    score_nd_parser.add_argument("-xS", "--merge_sys_text_gap", type=str, required=False, help="Merge system text gap character")
    score_nd_parser.add_argument("-aS", "--merge_sys_time_gap", type=str, required=False, help="Merge system time gap second")
    score_nd_parser.add_argument("-lS", "--combine_sys_llrs", type=str, choices=['min_llr', 'max_llr'], required=False, help="Choose min_llr or max_llr to combine system llrs for the system instances merging")
    score_nd_parser.add_argument("-vS", "--merge_sys_label", type=str, choices=['class', 'class-status'], required=False, help="Choose class or class-status to define how to handle the adhere/violate labels for the system instances merging. class is to use the class label only (ignoring status) to merge and class-status is to use the class and status label to merge")
    score_nd_parser.add_argument("-f", "--fix_ref_status_conflict", action='store_true', help="Change reference annotation to noann when there are the same annotations but different status")
    score_nd_parser.add_argument("-d", "--dump_inputs", action='store_true', help="Dump reference and system inputs as they are processed during scoring.")
    score_nd_parser.add_argument("-q", "--quiet", action='store_true', default=False, help="Do not dump ther final alignment and scores to stdout.")
    score_nd_parser.add_argument(      "--align_hacks", type=str, default="", help="NIST Field for undocumented experiments.")
    score_nd_parser.add_argument("-lf", "--llr_filter", type=str, default="", help="Filter system output by LLRs. The option requires the form <ORDER>:by_value:<VALUE>.  <ORDER> is one of: 'after_read|after_transforms'.  <VALUE> is the floating point threshold to retain detections with values >= <VALUE>")

    score_nd_parser.set_defaults(func=score_submission.score_nd_submission_dir_cli)

    score_ed_parser = subs.add_parser('score-ed', description='Score an emotion detection submission directory')
    score_ed_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing a emotion detection submission')
    score_ed_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
    score_ed_parser.add_argument('-i','--scoring-index-file', type=str, required=True, help='Use to filter file from scoring (REF)')
    score_ed_parser.add_argument('-e', '--emotion_list_file', type=str, required=False, help="Use to filter emotion from scoring (REF)")
    score_ed_parser.add_argument("-t", "--iou_thresholds", nargs='?', default="0.2", help="A comma separated list of IoU thresholds and the default value is 0.2.  Alternative criteria can be used by specifying the metric and its value using the form <METRIC>:<OPERATION>:<VALUE>.  'iou:gt:0.2' is the default value. The defined <METRIC>s are 'iou' and 'intersection'. The operations are 'gt' for '>', or 'gte' for '>='")
    score_ed_parser.add_argument("-aC", "--time_span_scale_collar", nargs='?', default="15", help="Forgiveness collar (in seconds) to determine true positive or false positive in scale metric alignment")
    score_ed_parser.add_argument("-xC", "--text_span_scale_collar", nargs='?', default="150", help="Forgiveness collar (in characters) to determine true positive or false positive in scale metric alignment")
    score_ed_parser.add_argument("-o", "--output_dir", type=str, nargs='?', default="tmp", help="Output directory")
    score_ed_parser.add_argument("-xR", "--merge_ref_text_gap", type=str, required=False, help="Merge reference text gap character")
    score_ed_parser.add_argument("-aR", "--merge_ref_time_gap", type=str, required=False, help="Merge reference time gap second")
    score_ed_parser.add_argument("-xS", "--merge_sys_text_gap", type=str, required=False, help="Merge system text gap character")
    score_ed_parser.add_argument("-aS", "--merge_sys_time_gap", type=str, required=False, help="Merge system time gap second")
    score_ed_parser.add_argument("-lS", "--combine_sys_llrs", type=str, choices=['min_llr', 'max_llr'], required=False, help="Choose min_llr or max_llr to combine system llrs for the system instances merging")
    score_ed_parser.add_argument("-vS", "--merge_sys_label", type=str, choices=['class'], required=False, help="Provide class only to define how to handle the status labels for the system instances merging. class is to use the class label only to merge")
    score_ed_parser.add_argument("-mv", "--minimum_vote_agreement", type=int, default=2, required=False, help="Set the mimimum agreement for voting between annotators. Default is 2 agreeing annotators per segment.")
    score_ed_parser.add_argument("-d", "--dump_inputs", action='store_true', help="Dump reference and system inputs as they are processed during scoring.")
    score_ed_parser.add_argument("-q", "--quiet", action='store_true', default=False, help="Do not dump ther final alignment and scores to stdout.")
    score_ed_parser.add_argument(      "--align_hacks", type=str, default="", help="Provide 'ManyRef:ManyHyp' to apply Many-to-Many instance matching alignment algorithm")
    score_ed_parser.add_argument("-lf", "--llr_filter", type=str, default="", help="Filter system output by LLRs.  The option requires the form <ORDER>:by_value:<VALUE>.  <ORDER> is one of: 'after_read|after_transforms'.  <VALUE> is the floating point threshold to retain detections with values >= <VALUE>")
    score_ed_parser.add_argument("-fm", "--file_merge_proportion", type=str, choices=['1/5', '1/4', '1/3', '1/2', '1'], required=False, help='choose one from list to define Filemerge proportion')

    score_ed_parser.set_defaults(func=score_submission.score_ed_submission_dir_cli)

    score_vd_parser = subs.add_parser('score-vd', description='Score a valence diarization submission directory')
    score_vd_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing a valence diarization submission')
    score_vd_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
    score_vd_parser.add_argument('-i','--scoring-index-file', type=str, required=True, help='Use to filter file from scoring (REF)')
    score_vd_parser.add_argument("-o", "--output_dir", type=str, nargs='?', default="tmp", help="Output directory")
    score_vd_parser.add_argument("-q", "--quiet", action='store_true', default=False, help="Do not dump ther final alignment and scores to stdout.")
    score_vd_parser.add_argument("-g", "--gap-allowed", action='store_true', default=False, help="Allow gap in system output of valence and arousal")

    score_vd_parser.set_defaults(func=score_submission.score_vd_submission_dir_cli)

    score_ad_parser = subs.add_parser('score-ad', description='Score an arousal diarization submission directory')
    score_ad_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing an arousal diarization submission')
    score_ad_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
    score_ad_parser.add_argument('-i','--scoring-index-file', type=str, required=True, help='Use to filter file from scoring (REF)')
    score_ad_parser.add_argument("-o", "--output_dir", type=str, nargs='?', default="tmp", help="Output directory")
    score_ad_parser.add_argument("-q", "--quiet", action='store_true', default=False, help="Do not dump ther final alignment and scores to stdout.")
    score_ad_parser.add_argument("-g", "--gap-allowed", action='store_true', default=False, help="Allow gap in system output of valence and arousal")

    score_ad_parser.set_defaults(func=score_submission.score_ad_submission_dir_cli)

    score_cd_parser = subs.add_parser('score-cd', description='Score a change detection submission directory')
    score_cd_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing a change detection submission')
    score_cd_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
    score_cd_parser.add_argument('-i','--scoring-index-file', type=str, required=True, help='Use to filter file from scoring (REF)')
    score_cd_parser.add_argument("-e", "--delta_cp_text_thresholds", nargs='?', default="100", help="A comma separated list of delta CP text thresholds.")
    score_cd_parser.add_argument("-m", "--delta_cp_time_thresholds", nargs='?', default="10", help="A comma separated list of delta CP time thresholds.")
    score_cd_parser.add_argument("-o", "--output_dir", type=str, nargs='?', default="tmp", help="Output directory")
    score_cd_parser.add_argument("-d", "--dump_inputs", action='store_true', help="Dump reference and system inputs as they are processed during scoring.")
    score_cd_parser.add_argument("-q", "--quiet", action='store_true', default=False, help="Do not dump ther final alignment and scores to stdout.")
    score_cd_parser.add_argument("-lf", "--llr_filter", type=str, default="", help="Filter system output by LLRs.  The option requires the form <ORDER>:by_value:<VALUE>.  <ORDER> is one of: 'after_read|after_transforms'.  <VALUE> is the floating point threshold to retain detections with values >= <VALUE>")

    score_cd_parser.set_defaults(func=score_submission.score_cd_submission_dir_cli)


    args = parser.parse_args()

    if not check_valid_argument_pair_merge_sys(args):
        if "score_nd" in str(vars(args)['func']):
            score_nd_parser.error('The -xS or -aS argument requires the -lS and -vS')
        if "score_ed" in str(vars(args)['func']):
            score_ed_parser.error('The -xS or -aS argument requires the -lS and -vS')

    if not check_valid_argument_pair_merge_ref(args):
        if "score_nd" in str(vars(args)['func']):
            score_nd_parser.error('The -xR or -aR argument requires the -vR')
            
    if "score_nd" in str(vars(args)['func']) or "score_ed" in str(vars(args)['func']):
        ### This is to make sure the argument is parsable before anything is done.  The result is ignored for now.  The command throws an assertion error to exit
        o = (score_submission.parse_thresholds(args.iou_thresholds))

    if "score_nd" in str(vars(args)['func']) or "score_ed" in str(vars(args)['func']) or "score_cd" in str(vars(args)['func']):
        ### This is to make sure the argument is parsable before anything is done.  The result is ignored for now.  The command throws an assertion error to exit
        o = (score_submission.parse_llr_filter(args.llr_filter))
        
    if hasattr(args, 'func') and args.func:    
        args.func(args)
    else:
        parser.print_help()
    
if __name__ == '__main__':
    main()
