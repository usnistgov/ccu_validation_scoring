# Computational Cultural Understanding (CCU) Evaluation Scoring Toolkit

**Version:** 1.0.0

**Date:** October 7th, 2022


## Table of Content

[Overview](#overview)

[Setup](#setup)

[Usage](#usage)

[Report a Bug](#contacts)

[Authors](#authors)

[Licensing Statement](#license)

## <a name="overview">Overview</a>


This package contains the tools to validate and score the TA1 evaluation tasks ND (norm discovery), ED (emotion detection), VD (valence diarization), AD (arousal diarization), CD (change detection) and scoring tools for the Hidden Norms (NDMAP). Please refer to the CCU evaluation plan for more information about CCU, the evaluation tasks, and file formats

This README file describes the reference annotation validation tool, system output validation tools and scoring tools.

 - Reference Validation Tool: confirms that a reference annotation set follows the rules set in the CCU Evaluation Plan.
 - System Output Validation Tool: confirms that a submission of system output follows the rules set in the CCU Evaluation Plan.
 - Scoring Tool: scores a system output submission against a reference with a scoring index file.



## <a name="setup">Setup</a>

The tools mentioned above are included in a Python package. They can be run under a shell terminal and have been confirmed to work under OS X and Linux.



### <a name="prerequisites">Prerequisites</a>

- [Python >= 3.8.6](https://www.python.org/downloads/)
- [Pandas >= 1.4.2](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
- [Pathlib >= 1.0.1](https://pypi.org/project/pathlib/)
- [Numpy >= 1.22.3](https://numpy.org/install/)
- [Pytest >= 7.1.3](https://docs.pytest.org/en/7.1.x/getting-started.html)


### <a name="installation">Installation</a>

Install the Python package using the following commands:

```bash
tar -xvzf /path/to/CCU_validation_scoring-x.x.x.tgz

cd /path/to/CCU_validation_scoring-x.x.x

python3 -m pip install -e ./
```

## <a name="usage">Usage</a>

In the `CCU_validation_scoring-x.x.x/` directory, run the following to get the version and usage:

```bash
CCU_scoring version

```
CCU_scoring is an umbrella tool that has many subcommands, each with its on set of command line options.  To get a list of subcommands, execute the command:

```bash
CCU_scoring -h
```

To get the command line options for a specific subcommand, use the subcommand `-h` flag. For example,

```bash
CCU_scoring score-nd -h
```

### Reference Validation Subcommand

**Validate a reference annotation** directory to make sure the files under the `reference_directory` have the required files in the correct format. The reference directory must follow the LDC annotation package directory structure.


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

Each evaluation task has a subcommand to validate a system output file.  The evaluation task include Norm Discovery, Emotion Detection, Valence Diarization, Arousal Diarization, Change Detection.  To **validate the format of a nd/ed/vd/ad/cd submission** directory against a reference directory with a scoring index file use the commands:

```bash
CCU_scoring validate-nd -s <submission_directory> -ref <reference_directory> -i <scoring_index_file>
CCU_scoring validate-ed -s <submission_directory> -ref <reference_directory> -i <scoring_index_file>
CCU_scoring validate-vd -s <submission_directory> -ref <reference_directory> -i <scoring_index_file>
CCU_scoring validate-ad -s <submission_directory> -ref <reference_directory> -i <scoring_index_file>
CCU_scoring validate-cd -s <submission_directory> -ref <reference_directory> -i <scoring_index_file>
```

**Required Arguments**

 * `-s`: submission directory

 * `-ref`: reference directory

 * `-i`: file containing the list of documents to score

```bash
# an example of submission validation
CCU_scoring validate-nd \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220815_164235 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab
```

**Norm Discovery Mapping Validation**

To **validate the format of an NDMAP submission** directory against a reference directory with a scoring index file, use the command below.  This validation only applies to the mapping file, not the original system.

```bash
CCU_scoring validate-ndmap -s <submission_directory> -n <hidden_norm_list_file>
```

**Required Arguments**

 * `-s`: submission directory

 * `-n`: file containing the hidden norm

```bash
# an example of ndmap submission validation
CCU_scoring validate-ndmap \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/NDMAP/CCU_P1_TA1_NDMAP_NIST_mini-eval1_20220605_050236 \
-n test/hidden_norms.txt 
```

### Submission Scoring Subcommands

**Norm Discovery (ND) Scoring Subcommand**

To **score an ND submission** directory against a reference directory with a scoring index file, us the command:

```bash
CCU_scoring score-nd -s <norm_submission_directory> -ref <reference_directory> -i <scoring_index_file>
```

**Norm Discovery Mapping Scoring Subcommand**

To **score an NDMAP submission** directory and an ND submission using against a reference directory with a scoring index file, use the command:

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

```bash
# an example of norm scoring
CCU_scoring score-nd \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220815_164235 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab

# an example of ndmap scoring
CCU_scoring score-nd \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220531_050236 \
-m test/pass_submissions/pass_submissions_LDC_reference_sample/NDMAP/CCU_P1_TA1_NDMAP_NIST_mini-eval1_20220605_050236 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab
```

**Emotion Detection Scoring**

To **score an ED submission** directory against a reference directory with a scoring index file:

```bash
CCU_scoring score-ed -s <norm_submission_directory> -m <norm_mapping_submission_directory> -ref <reference_directory> -i <scoring_index_file>
```

**Required Arguments**

 * `-s`: emotion submission directory

 * `-ref`: reference directory

 * `-i`: file containing the file id of scoring datasets

**Optional Arguments**

 * `-e`: file containing the emotion to filter emotion from scoring

 * `-t`: comma separated list of IoU thresholds

 * `-o`: output directory containing the score and alignment file

```bash
# an example of ed scoring
CCU_scoring score-ed \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/ED/CCU_P1_TA1_ED_NIST_mini-eval1_20220531_050236 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab
```

**Valence Detection (VD) and Arousal Detection (AD) Scoring Subcommands**

To **score an VD or AD submission** directory against a reference directory with a scoring index file:

```bash
CCU_scoring score-vd -s <norm_submission_directory> -m <norm_mapping_submission_directory> -ref <reference_directory> -i <scoring_index_file>
CCU_scoring score-ad -s <norm_submission_directory> -m <norm_mapping_submission_directory> -ref <reference_directory> -i <scoring_index_file>
```

**Required Arguments**

 * `-s`: emotion submission directory

 * `-ref`: reference directory

 * `-i`: file containing the file id of scoring datasets

**Optional Arguments**

 * `-o`: output directory containing the score and diarization file

```bash
# an example of vd scoring
CCU_scoring score-vd \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/VD/CCU_P1_TA1_VD_NIST_mini-eval1_20220531_050236 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab

# an example of ad scoring
CCU_scoring score-ad \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/AD/CCU_P1_TA1_AD_NIST_mini-eval1_20220531_050236 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab
```

**Change Detection (CD) Scoring**

To **score a CD submission** directory against a reference directory with a scoring index file:

```bash
CCU_scoring score-cd -s <norm_submission_directory> -m <norm_mapping_submission_directory> -ref <reference_directory> -i <scoring_index_file>
```

**Required Arguments**

 * `-s`: emotion submission directory

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
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab 
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