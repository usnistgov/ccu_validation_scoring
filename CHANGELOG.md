# CHANGELOG
All notable changes to this project will be documented in this file.

## [1.1.1] - 2023-03-03
- Matplotlib is a required package
- Added Instance_alignment.tab derivative outputs for ND, ED, and CD:
  - Instance counts for instance_alignment_class_stats.tab
  - Histograms of instance IoU, Collar-Based IoU, and LLRs. See 'instance_alignment_grqphs.png'
  - A confusion count matrics for norm status.  See 'instance_alignment_status_confusion.tab'
- Added Precision/Recall Curves for the ND, ED, and CD scorer.  See output files 'pr_*png'.'
- Changed the ND reference merge algorithm to ignore or use the norm status via the -vR option.
- Extended the ND and ED correctness contraints to use additional metrics like intersection.  See '-t'.
- Modified the instance alignment files:
  - Added ref and system norm status fields
  - Added more values to the parameter field for ND and ED. intersection, union, etc.
- Added the -d option for ND and ED scores to dump the reference and system as 'read' in and as 'scored' after applying transformations.

## [1.1.0] - 2023-02-15
### Added
- Add four arguments to ND and ED scoring for reference and system merging
- Add more test cases to test the reference and system merging
- Add scoring_parameters.tab to output directory
- Add two statistic results (statistic_aggregated.tab and statistic_by_class.tab) to output directory

### Updated
- Fixed the calculation of AP when system have duplicate llrs.  This is a minor score changes and affects systems with high frequency common LLRs.
- Fixed the change detection scorer to exclude system detections in no score regions as defined by the segments annotation.
- Fixed the norm and emotion detection scorre to include false alarms that have no overlap with reference annotations.

## [1.0.1] - 2022-11-10
### Added
- Add a script to compute basic statistics on the reference data for norms or emotions task.
- Add a script to generate a random submission for norms or emotions task.
- Fix a bug of the scorer when all system output of specific norm/emotion were mapping to NO_SCORE_REGION of reference
- Remove scoring index option of validation command
- Use system_input index/file_info rather than reference to validate submission
- Fix the testcases based on the new validation rule
- Remove EMPTY_TBD string from annotation
- Update REAMDE
- Add CHANGELOG
