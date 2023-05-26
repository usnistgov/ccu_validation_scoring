# CHANGELOG
All notable changes to this project will be documented in this file.

## Upcoming changes
- Add the scaled IoU values and a histogram in instance_alignment_grqphs.png

## [1.3.0] - 2023-04-26
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
  - Added a new output graph 'instance_alignment_grqphs.png' that plots the distribution of LLRs and IoUs.
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
