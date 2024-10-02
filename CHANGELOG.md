# CHANGELOG
All notable changes to this project will be documented in this file.

## [1.3.5] - 2024-10-02
- Implemented full, half, one third and quarter filemerge scoring method for ED scoring
- Added judgement voting threshold for ED scoring. Default is still 2
- Added LLR filtering to filter system output for CD, ND and ED scoring
- Added micro metrics for ND and ED scoring

## [1.3.4] - 2024-03-14
- Added OpenCCU testcase and readme
- Updated the format of changepoint to let it will be compatible with the previous CCU evaluation
- Updated the script that generate perfect submission to include OpenCCU format
- Added a optional argument to ND scoring command to fix reference status conflict

## [1.3.3] - 2023-12-15
- Updated data type in function sumup_tad_class_level_scores to be compatible with pandas 2.1.3 version
- Fixed syntax error in function call in function generate_zero_scores_norm_emotion
- Updated minimum pandas version requirement to 2.0.3 to remove deprecation warning for pytest

## [1.3.2] - 2023-12-14
- Important: Added a new requirement for norm validation. Norm string must be either in LDC defined norms or system generated norms that start with 5
- Updated the format of changepoint reference and system output in validation and scoring
- Added zscore normalization to valence/aroual reference and system output before scoring
- Fixed duration calculation in text format for norm/emotion scoring

## [1.3.1] - 2023-09-06
- Optimized the process of generating alignment for norm/emotion scoring
- Added a optional parameter to allow gap in sys for valence/aroual validation. This change was for TA2 logs
- Added no score region to sys if there are gaps in sys for valence/aroual scoring. This change was for TA2 logs
- Added coverage metric to valence/aroual result
- Added a warning and remove the reference segment in scoring if start is the same as end in audio/video format

## [1.3.0] - 2023-05-26
- Modified the handling of pre/post annotation NoScore regions.  The old code would drop system instances that span the entire file.
- Added the  --align_hacks ManyRef:OneHyp option for ND and ED.
- minor fixes to handle scoring Eval1-LC1.

## [1.2.4] - 2023-04-21
- Corrected handling of single LLR ND and ED systems found in the minieval.
- Added source type to the ED and ND alignmenrs.

## [1.2.3] - 2023-04-17
- Corrected an incorrect repair to the scaled FP/TP/F1 calculations.  Some alignment records were not included.

## [1.2.2] - 2023-04-11
- Set the default score for a no system output (or reference) to be 0 for Precision/Recalls/F1.

## [1.2.1] - 2023-04-04
- The scaled F1 scoring was corrected to replicate the initial scoring sent to teams.
- Various additiona minor fixes.

## [1.2.0] - 2023-03-23
- Major change for the norm and emotion score:
  - The scaled F1 measures were added.
  - The --iou_thresholds option value was generalized to support 'greater than' operations for checks.
  - Added --time_span_scale_collar and --text_span_scale_collar options for setting the scaled F1 measures.
  - Added a new output graph 'instance_alignment_graphs.png' that plots the distribution of LLRs and IoUs.
  - Added a new output graph 'instance_alignment_status_confusion.tab' that reports norm 'status' confusions.
  - The instant_alingment.tab parameter column has additional values for the scaled F1 calculations. 
  - Added the -d option for ND and ED scorers to dump the input ref and sys dataframes and the converted ref and sys
    dataframes used for scoring (after all filtering/conversions).
  - Changes to scores_by_class.tab and scores_aggregated.tab output files.  The major changes included:
    - Adding measurements taking at the 'Minimum LLR' which includes All system output produced by the system.
      These values using the metric name '_at_MinLLR' in the files.
    - Duplicated the AP values in scores_by_class.tab naming the metric 'average_precision'
    - Duplicated the mAP values in scores_aggregated.tab naming the metric 'mean_average_precision'
    - The scores_aggregated.tab file uses the convention of pre-pending 'mean_' to represent aggregated means.
    - Added values for the following measures in the scores_by_class.tab file and their class means in
      scores_aggregated.tabs:
      - average_precision
      - recall_at_MinLLR
      - precision_at_MinLLR
      - f1_at_MinLLR
      - llr_at_MinLLR
      - recall_at_MinLLR
      - scaled_f1_at_MinLLR
      - scaled_precision_at_MinLLR
      - scaled_recall_at_MinLLR
      - sum_fp_at_MinLLR
      - sum_scaled_fp_at_MinLLR
      - sum_scaled_tp_at_MinLLR
      - sum_tp_at_MinLLR
      - scaled_recall_at_MinLLR

## [1.1.1] - 2023-03-03
- Matplotlib is a required package
- Added Instance_alignment.tab derivative outputs for ND, ED, and CD:
  - Instance counts for instance_alignment_class_stats.tab
  - Histograms of instance IoU, Collar-Based IoU, and LLRs. See 'instance_alignment_graphs.png'
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
- Added four arguments to ND and ED scoring for reference and system merging
- Added more test cases to test the reference and system merging
- Added scoring_parameters.tab to output directory
- Added two statistic results (statistic_aggregated.tab and statistic_by_class.tab) to output directory

### Updated
- Fixed the calculation of AP when system have duplicate llrs.  This is a minor score changes and affects systems with high frequency common LLRs.
- Fixed the change detection scorer to exclude system detections in no score regions as defined by the segments annotation.
- Fixed the norm and emotion detection scorre to include false alarms that have no overlap with reference annotations.

## [1.0.1] - 2022-11-10
### Added
- Added a script to compute basic statistics on the reference data for norms or emotions task.
- Added a script to generate a random submission for norms or emotions task.
- Fixed a bug of the scorer when all system output of specific norm/emotion were mapping to NO_SCORE_REGION of reference
- Removed scoring index option of validation command
- Used system_input index/file_info rather than reference to validate submission
- Fixed the testcases based on the new validation rule
- Removed EMPTY_TBD string from annotation
- Updated REAMDE
- Added CHANGELOG
