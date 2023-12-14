# Computational Cultural Understanding (CCU) Evaluation Validation and Scoring Toolkit

**Version:** 1.3.0

**Date:** May 26, 2023


## Table of Content

[Overview](#overview)

[Setup](#setup)

[Directory Structure and File Format](#format)

[Usage](#usage)

[Report a Bug](#contacts)

[Authors](#authors)

[Licensing Statement](#license)

## <a name="overview">Overview</a>


This package contains the tools to validate and score the TA1 evaluation tasks: ND (norm discovery), ED (emotion detection), VD (valence diarization), AD (arousal diarization), CD (change detection) and scoring tools for the Hidden Norms (NDMAP). Please refer to the CCU Evaluation Plan for more information about CCU, the evaluation tasks, and the file formats.

This README file describes the reference annotation validation tool, system output validation tool, scoring tool, reference statistics computing tool and random submission generation tool. For the README of OpenCCU, please click and read [this](OpenCCU_README.md) document.

 - Reference Validation Tool: confirms that a reference annotation set follows the LDC CCU annotation package directory structure.
 - System Output Validation Tool: confirms that a submission of system output follows the rules set in the CCU Evaluation Plan.
 - Scoring Tool: scores a system output submission against a reference with a scoring index file.
 - Reference Statistics Computing Tool: computes basic statistics on the reference data for the ND or ED task.
 - Random Submission Generation Tool: generates a random submission for the ND or ED task.


## <a name="setup">Setup</a>

The tools mentioned above are included as a Python package. They can be run under a shell terminal and have been confirmed to work under OS X and Ubuntu.



### <a name="prerequisites">Prerequisites</a>

- [Python >= 3.8.6](https://www.python.org/downloads/)
- [Pandas >= 1.4.2](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
- [Pathlib >= 1.0.1](https://pypi.org/project/pathlib/)
- [Numpy >= 1.22.3](https://numpy.org/install/)
- [Pytest >= 7.1.3](https://docs.pytest.org/en/7.1.x/getting-started.html)
- [matplotlib >= 3.5.2](https://matplotlib.org/stable/users/getting_started/index.html#installation-quick-start)

### <a name="installation">Installation</a>

Install the Python package using the following commands:

```bash
git clone https://github.com/usnistgov/ccu_validation_scoring

cd ./CCU_validation_scoring

python3 -m pip install -e ./
```

## <a name="format">Directory Structure and File Format</a>
The CCU validation and scoring toolkit expects input directories and/or files to have specific structures and formats. This section gives more information on these structures and formats that are referred in subsequent sections. 

The `reference directory` mentioned validation and scoring sections must follow the LDC annotation data package directory structure and at a minimum must contain the following files in the given directory structure to pass validation:

```bash
<reference_directory>/
     ./data/
          norms.tab
          emotions.tab
          valence_arousal.tab
          changepoint.tab
     ./docs/
          segments.tab
          file_info.tab
     ./index_files/
          <DATASET>.system_input.index.tab
```

where `<DATASET>` is the name of dataset.

Please refer to the LDC CCU annotation data package `README` for the formats of the above `.tab`
files. 

The toolkit includes several sample reference datasets for testing. See `ccu_validation_scoring/test/reference/LDC_reference_sample` or other sibling directories.

The toolkit uses different index files for various purposes:

* `system input index file` - tells the scorer which files are available for the system to process. This file is included in the CCU source data package and is used in validation and generation of a sample submission. The format is described in the CCU Evaluation Plan.
* `system output index file` - tells the scorer which files were processed by the system. This file is generated by the users and is used in validation and scoring. It must be located inside the `submission directory`. The format is described in the CCU Evaluation Plan.
* `scoring index file` - tells the scorer which files to score to facilitate subset scoring. This file is generated by the users and is used in scoring. The `scoring index file` has one column with the header `file_id` containing file IDs to score, one per row.

An example of a `system input index file` can be found in the sample reference datasets:
```bash
ccu_validation_scoring/test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.system_input.index.tab
```

An example of a `system output index file` can be found in the sample submissions:
```bash
ccu_validation_scoring/test/pass_submissions/pass_submissions_LDC_
reference_sample/ED/CCU_P1_TA1_ED_NIST_mini-eval1_20220816_050236/system_output.index.tab
```

## <a name="usage">Usage</a>

In the `CCU_validation_scoring-x.x.x/` directory, run the following to get the version and usage:

```bash
CCU_scoring version

```
CCU_scoring is an umbrella tool that has several subcommands, each with its own set of command line options.  To get a list of subcommands, execute:

```bash
CCU_scoring -h
```

Use the `-h` flag on the subcommand to get the subcommand help manual. For example:

```bash
CCU_scoring score-nd -h
```

### Reference Validation Subcommand

**Validate a reference annotation directory** to make sure the `reference directory` have the required files.

```bash
CCU_scoring validate-ref -ref <reference_directory>
```

**Required Arguments**

 * `-ref`: reference directory

```bash
# an example of reference validation
CCU_scoring validate-ref -ref test/reference/LDC_reference_sample
```

### Submission Validation Subcommands

Each evaluation task has a subcommand to validate a system output file.  The evaluation tasks include Norm Discovery (ND), Emotion Detection (ED), Valence Diarization (VD), Arousal Diarization (AD), and Change Detection (CD).  Use the subcommands below to **validate the format of a ND/ED/VD/AD/CD submission directory** against a `reference directory`. The `submission directory` must include a `system output index file`. 

```bash
CCU_scoring validate-nd -s <submission_directory> -ref <reference_directory>
CCU_scoring validate-ed -s <submission_directory> -ref <reference_directory> 
CCU_scoring validate-vd -s <submission_directory> -ref <reference_directory>
CCU_scoring validate-ad -s <submission_directory> -ref <reference_directory>
CCU_scoring validate-cd -s <submission_directory> -ref <reference_directory>
```

**Required Arguments**

 * `-s`: submission directory

 * `-ref`: reference directory

```bash
# an example of submission validation
CCU_scoring validate-nd \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220815_164235 \
-ref test/reference/LDC_reference_sample
```

**Norm Discovery Mapping Validation**

Use the command below to **validate the format of an NDMAP submission directory** with a `hidden norm list`.  This validation only applies to the mapping file, not the original system. The `hidden norm list` has one column (no header) containing norm IDs, one per row.

```bash
CCU_scoring validate-ndmap -s <submission_directory> -n <hidden_norm_list_file>
```

**Required Arguments**

 * `-s`: submission directory

 * `-n`: file containing the hidden norm mapping

```bash
# an example of ndmap submission validation
CCU_scoring validate-ndmap \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/NDMAP/CCU_P1_TA1_NDMAP_NIST_mini-eval1_20220605_050236 \
-n test/hidden_norms.txt 
```

### Submission Scoring Subcommands

**Norm Discovery (ND) Scoring Subcommand**

Use the command below to **score an ND submission directory** against a `reference directory` with a `scoring index file`. The `submission directory` must include a `system output index file`. 

```bash
CCU_scoring score-nd -s <norm_submission_directory> -ref <reference_directory> -i <scoring_index_file>
```

**Norm Discovery Mapping Scoring Subcommand**

Use the command below to **score an NDMAP submission directory** and an ND submission against a `reference directory` with a `scoring index file`. The `submission directory` must include a `system output index file`. 

```bash
CCU_scoring score-nd -s <norm_submission_directory> -m <norm_mapping_submission_directory> -ref <reference_directory> -i <scoring_index_file>
```

**Required Arguments**

 * `-s`: norm submission directory

 * `-ref`: reference directory

 * `-i`: file containing the file id of scoring datasets

**Optional Arguments**

 * `-m`: norm mapping submission directory

 * `-n`: file containing the norm to filter norm from scoring

 * `-t`: comma separated list of IoU thresholds

 * `-o`: output directory containing the score and alignment file

 * `-xR`: character gap for the text reference instances merging

 * `-aR`: second gap for the time reference instances merging

 * `-xS`: character gap for the text system instances merging

 * `-aS`: second gap for the time system instances merging

 * `-lS`: choose min_llr or max_llr to combine system llrs for the system instances merging

 * `-vS`: choose class or class-status to define how to handle the adhere/violate labels for the system instances merging. class is to use the class label only to merge and class-status is to use the class and status label to merge

```bash
# an example of norm scoring
CCU_scoring score-nd \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220815_164235 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab

# an example of ndmap scoring
CCU_scoring score-nd \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220531_050236 \
-m test/pass_submissions/pass_submissions_LDC_reference_sample/NDMAP/CCU_P1_TA1_NDMAP_NIST_mini-eval1_20220605_050236 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab
```

**Emotion Detection (ED) Scoring Subcommand**

Use the command below to **score an ED submission directory** against a `reference directory` with a `scoring index file`. The `submission directory` must include a `system output index file`. 

```bash
CCU_scoring score-ed -s <emotion_submission_directory> -ref <reference_directory> -i <scoring_index_file>
```

**Required Arguments**

 * `-s`: emotion submission directory

 * `-ref`: reference directory

 * `-i`: file containing the file id of scoring datasets

**Optional Arguments**

 * `-e`: file containing the emotion to filter emotion from scoring

 * `-t`: comma separated list of IoU thresholds

 * `-o`: output directory containing the score and alignment file

 * `-xR`: character gap for the text reference instances merging

 * `-aR`: second gap for the time reference instances merging

 * `-xS`: character gap for the text system instances merging

 * `-aS`: second gap for the time system instances merging

 * `-lS`: choose min_llr or max_llr to combine system llrs for the system instances merging

 * `-vS`: choose class or class-status to define how to handle the adhere/violate labels for the system instances merging. class is to use the class label only to merge and class-status is to use the class and status label to merge

```bash
# an example of ed scoring
CCU_scoring score-ed \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/ED/CCU_P1_TA1_ED_NIST_mini-eval1_20220531_050236 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.ED.scoring.index.tab
```

**Valence Detection (VD) and Arousal Detection (AD) Scoring Subcommands**

Use the commands below to **score an VD or AD submission directory** against a `reference directory` with a `scoring index file`. The `submission directory` must include a `system output index file`. 

```bash
CCU_scoring score-vd -s <valence_submission_directory> -ref <reference_directory> -i <scoring_index_file>
CCU_scoring score-ad -s <arousal_submission_directory> -ref <reference_directory> -i <scoring_index_file>
```

**Required Arguments**

 * `-s`: valence or arousal submission directory

 * `-ref`: reference directory

 * `-i`: file containing the file id of scoring datasets

**Optional Arguments**

 * `-o`: output directory containing the score and diarization file

```bash
# an example of vd scoring
CCU_scoring score-vd \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/VD/CCU_P1_TA1_VD_NIST_mini-eval1_20220531_050236 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.VD.scoring.index.tab

# an example of ad scoring
CCU_scoring score-ad \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/AD/CCU_P1_TA1_AD_NIST_mini-eval1_20220531_050236 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.AD.scoring.index.tab
```

**Change Detection (CD) Scoring Subcommand**

Use the command below to **score a CD submission directory** against a `reference directory` with a `scoring index file`. The `submission directory` must include a `system output index file`. 

```bash
CCU_scoring score-cd -s <change_submission_directory> -ref <reference_directory> -i <scoring_index_file>
```

**Required Arguments**

 * `-s`: change detection submission directory

 * `-ref`: reference directory

 * `-i`: file containing the file id of scoring datasets

**Optional Arguments**

 * `-e`: comma separated list of delta CP text thresholds

 * `-m`: comma separated list of delta CP time thresholds

 * `-o`: output directory containing the score and alignment file

```bash
# an example of cd scoring
CCU_scoring score-cd \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/CD/CCU_P1_TA1_CD_NIST_mini-eval1_20220531_050236 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.CD.scoring.index.tab 
```
### Reference Statistics Computing Tool

The following command should be run within the `CCU_validation_scoring-x.x.x/` directory.

```bash
python3 scripts/ccu_ref_analysis.py -r <reference_directory> -t <task_string> -i <scoring_index_file> -o <output_file>
```

**Required Arguments**

 * `-r`: reference directory
 * `-t`: norms or emotions
 * `-i`: file containing the file id of scoring datasets
 * `-o`: file where the statistics will be output

```bash
# an example of statistics computing
python3 scripts/ccu_ref_analysis.py -r test/reference/LDC_reference_sample \
-t norms \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab \
-o tmp.tab
```

### Random Submission Generation Tool

The following command should be run within the `CCU_validation_scoring-x.x.x/` directory.

```bash
python3 scripts/generate_random_submission.py -ref <reference_directory> -t <task_string> -i <scoring_index_file> -o <output_directory>
```

**Required Arguments**

 * `-ref`: reference directory
 * `-t`: norms or emotions
 * `-i`: file containing the file id of scoring datasets
 * `-o`: output directory containing a random submission

```bash
# an example of statistics computing
python3 scripts/generate_random_submission.py -ref test/reference/LDC_reference_sample \
-t norms \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.ND.scoring.index.tab \
-o tmp
```

## <a name="contacts">Report a Bug</a>

Please send bug reports to [nist_ccu@nist.gov](mailto:nist_ccu@nist.gov)

For the bug report to be useful, please include the command line, files and text output, including the error message in your email.

### <a name="bugreport">Test case bug report</a>

A test suite has been developed and is runnable using the following command within the `CCU_validation_scoring-x.x.x/` directory:

This will run the tests against a set of submissions and reference files available under `test`.

```bash
pytest
```

## <a name="authors">Authors</a>

Jennifer Yu &lt;yan.yu@nist.gov&gt;

Clyburn Cunningham &lt;clyburn.cunningham@nist.gov&gt;

Lukas Diduch &lt;lukas.diduch@nist.gov&gt;

Jonathan Fiscus &lt;jonathan.fiscus@nist.gov&gt;

Audrey Tong &lt;audrey.tong@nist.gov&gt;

## <a name="license">Licensing Statement</a>

Full details can be found at: http://nist.gov/data/license.cfm

```
NIST-developed software is provided by NIST as a public service. You may use,
copy, and distribute copies of the software in any medium, provided that you
keep intact this entire notice. You may improve, modify, and create derivative
works of the software or any portion of the software, and you may copy and
distribute such modifications or works. Modified works should carry a notice
stating that you changed the software and should note the date and nature of
any such change. Please explicitly acknowledge the National Institute of
Standards and Technology as the source of the software. 

NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY
OF ANY KIND, EXPRESS, IMPLIED, IN FACT, OR ARISING BY OPERATION OF LAW,
INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, AND DATA ACCURACY. NIST NEITHER
REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE
UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES
NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR
THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY,
RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

You are solely responsible for determining the appropriateness of using and
distributing the software and you assume all risks associated with its use,
including but not limited to the risks and costs of program errors, compliance
with applicable laws, damage to or loss of data, programs or equipment, and the
unavailability or interruption of operation. This software is not intended to
be used in any situation where a failure could cause risk of injury or damage
to property. The software developed by NIST employees is not subject to
copyright protection within the United States.
```
