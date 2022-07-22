import argparse
import logging

from . import validate_submission

def print_package_version(args):
    print(__import__(__package__).__version__)

def main():
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(title='subcommands')

    # Add version subcommand
    version_parser = subs.add_parser('version', description='Print package version')
    version_parser.set_defaults(func=print_package_version)

    validate_nd_parser = subs.add_parser('validate-nd', description='Validate a norm detection submission directory')
    validate_nd_parser.add_argument('-s','--submission-dir', type=str, required=True, nargs=1, help='Directory containing a submission')
    validate_nd_parser.add_argument('-ref','--reference-dir', type=str, required=True, nargs=1, help='Reference directory')

    validate_nd_parser.set_defaults(func=validate_submission.validate_nd_submission_dir_cli)

    validate_ed_parser = subs.add_parser('validate-ed', description='Validate a submission directory')
    validate_ed_parser.add_argument('-s','--submission-dir', type=str, required=True, nargs=1, help='Directory containing a submission')
    validate_ed_parser.add_argument('-ref','--reference-dir', type=str, required=True, nargs=1, help='Reference directory')

    validate_ed_parser.set_defaults(func=validate_submission.validate_ed_submission_dir_cli)

    validate_vd_parser = subs.add_parser('validate-vd', description='Validate a submission directory')
    validate_vd_parser.add_argument('-s','--submission-dir', type=str, required=True, nargs=1, help='Directory containing a submission')
    validate_vd_parser.add_argument('-ref','--reference-dir', type=str, required=True, nargs=1, help='Reference directory')

    validate_vd_parser.set_defaults(func=validate_submission.validate_vd_submission_dir_cli)

    validate_ad_parser = subs.add_parser('validate_ad', description='Validate a submission directory')
    validate_ad_parser.add_argument('-s','--submission-dir', type=str, required=True, nargs=1, help='Directory containing a submission')
    validate_ad_parser.add_argument('-ref','--reference-dir', type=str, required=True, nargs=1, help='Reference directory')

    validate_ad_parser.set_defaults(func=validate_submission.validate_ad_submission_dir_cli)

    validate_cd_parser = subs.add_parser('validate_cd', description='Validate a submission directory')
    validate_cd_parser.add_argument('-s','--submission-dir', type=str, required=True, nargs=1, help='Directory containing a submission')
    validate_cd_parser.add_argument('-ref','--reference-dir', type=str, required=True, nargs=1, help='Reference directory')

    validate_cd_parser.set_defaults(func=validate_submission.validate_cd_submission_dir_cli)

    args = parser.parse_args()

    if hasattr(args, 'func') and args.func:    
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
