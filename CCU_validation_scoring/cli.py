import argparse
import logging

from . import validate_submission
# from . import validate_reference
from . import score_submission

def print_package_version(args):
    print(__import__(__package__).__version__)

def main():
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(title='subcommands')

    # Add version subcommand
    version_parser = subs.add_parser('version', description='Print package version')
    version_parser.set_defaults(func=print_package_version)

    # validate_ref_parser = subs.add_parser('validate-ref', description='Validate the reference directory')
    # validate_ref_parser.add_argument('-ref','--reference-dir', type=str, required=True, nargs=1, help='Reference directory')

    # validate_ref_parser.set_defaults(func=validate_reference.validate_ref_submission_dir_cli)

    # validate_nd_parser = subs.add_parser('validate-nd', description='Validate a norm detection submission directory')
    # validate_nd_parser.add_argument('-s','--submission-dir', type=str, required=True, nargs=1, help='Directory containing a submission')
    # validate_nd_parser.add_argument('-ref','--reference-dir', type=str, required=True, nargs=1, help='Reference directory')

    # validate_nd_parser.set_defaults(func=validate_submission.validate_nd_submission_dir_cli)

    # validate_ed_parser = subs.add_parser('validate-ed', description='Validate a submission directory')
    # validate_ed_parser.add_argument('-s','--submission-dir', type=str, required=True, nargs=1, help='Directory containing a submission')
    # validate_ed_parser.add_argument('-ref','--reference-dir', type=str, required=True, nargs=1, help='Reference directory')

    # validate_ed_parser.set_defaults(func=validate_submission.validate_ed_submission_dir_cli)

    # validate_vd_parser = subs.add_parser('validate-vd', description='Validate a submission directory')
    # validate_vd_parser.add_argument('-s','--submission-dir', type=str, required=True, nargs=1, help='Directory containing a submission')
    # validate_vd_parser.add_argument('-ref','--reference-dir', type=str, required=True, nargs=1, help='Reference directory')

    # validate_vd_parser.set_defaults(func=validate_submission.validate_vd_submission_dir_cli)

    # validate_ad_parser = subs.add_parser('validate_ad', description='Validate a submission directory')
    # validate_ad_parser.add_argument('-s','--submission-dir', type=str, required=True, nargs=1, help='Directory containing a submission')
    # validate_ad_parser.add_argument('-ref','--reference-dir', type=str, required=True, nargs=1, help='Reference directory')

    # validate_ad_parser.set_defaults(func=validate_submission.validate_ad_submission_dir_cli)

    # validate_cd_parser = subs.add_parser('validate_cd', description='Validate a submission directory')
    # validate_cd_parser.add_argument('-s','--submission-dir', type=str, required=True, nargs=1, help='Directory containing a submission')
    # validate_cd_parser.add_argument('-ref','--reference-dir', type=str, required=True, nargs=1, help='Reference directory')

    # validate_cd_parser.set_defaults(func=validate_submission.validate_cd_submission_dir_cli)

    score_nd_parser = subs.add_parser('score-nd', description='Score a norm detection submission directory')
    score_nd_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing a norm submission')
    score_nd_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
    score_nd_parser.add_argument('-m','--mapping-submission-dir', type=str, help='Directory containing a norm mapping submission')
    score_nd_parser.add_argument("-i", "--iou_thresholds", nargs='?', default="0.2", help="A comma separated list of IoU thresholds.")
    score_nd_parser.add_argument("-o", "--output_dir", type=str, nargs='?', default="tmp", help="Output the system level and class level score to a directory")

    score_nd_parser.set_defaults(func=score_submission.score_nd_submission_dir_cli)

    score_ed_parser = subs.add_parser('score-ed', description='Score a emotion detection submission directory')
    score_ed_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing a emotion submission')
    score_ed_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
    score_ed_parser.add_argument("-i", "--iou_thresholds", nargs='?', default="0.2", help="A comma separated list of IoU thresholds.")
    score_ed_parser.add_argument("-o", "--output_dir", type=str, nargs='?', default="tmp", help="Output the system level and class level score to a directory")

    score_ed_parser.set_defaults(func=score_submission.score_ed_submission_dir_cli)

    score_vd_parser = subs.add_parser('score-vd', description='Score a valence detection submission directory')
    score_vd_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing a valence submission')
    score_vd_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
    score_vd_parser.add_argument("-o", "--output_dir", type=str, nargs='?', default="tmp", help="Output the system level score to a directory")

    score_vd_parser.set_defaults(func=score_submission.score_vd_submission_dir_cli)

    score_ad_parser = subs.add_parser('score-ad', description='Score an arousal submission directory')
    score_ad_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing an arousal submission')
    score_ad_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
    score_ad_parser.add_argument("-o", "--output_dir", type=str, nargs='?', default="tmp", help="Output the system level score to a directory")

    score_ad_parser.set_defaults(func=score_submission.score_ad_submission_dir_cli)

    # score_cd_parser = subs.add_parser('score-cd', description='Score a changepoint submission directory')
    # score_cd_parser.add_argument('-s','--submission-dir', type=str, required=True, help='Directory containing a changepoint submission')
    # score_cd_parser.add_argument('-ref','--reference-dir', type=str, required=True, help='Reference directory')
    # score_cd_parser.add_argument("-d", "--delta_cp_thresholds", nargs='?', default="0.2", help="A comma separated list of delta CP thresholds.")
    # score_cd_parser.add_argument("-o", "--output_dir", type=str, nargs='?', default="tmp", help="Output the system level score to a directory")

    # score_cd_parser.set_defaults(func=score_submission.score_cd_submission_dir_cli)


    args = parser.parse_args()

    if hasattr(args, 'func') and args.func:    
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
