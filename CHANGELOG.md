# CHANGELOG
All notable changes to this project will be documented in this file.

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

## [1.1.0] - 2023-02-02
### Added
- Add four arguments to ND and ED scoring for reference and system merging
- Add more test cases to test the reference and system merging
- Add scoring_parameters.tab to output directory
- Add two statistic results (statistic_aggregated.tab and statistic_by_class.tab) to output directory

### Updated
- Fix the calculation of AP when system have duplicate llrs
