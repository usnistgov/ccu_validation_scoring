# Computational Cultural Understanding (CCU) Toolkit

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


This directory contains the tools to validate and score several tasks (ND, NDMAP, ED, VD, AD, CD). There are used in the  **Computational Cultural Understanding** (CCU) evaluation.

The goal of the CCU program is to create human language technologies that will provide effective dialogue assistance to monolingual operators in cross-cultural interactions.

This README file describes a reference validation tool, six submission validation tools and five submission scoring tools

 - Reference Validation Tool: confirms that a reference follows the rules set in the CCU Evaluation Plan.
 - Submission Validation Tool: confirms that a submission follows the rules set in the CCU Evaluation Plan.
 - Submission Scoring Tool: scores a submission against a reference with a scoring index file.



## <a name="setup">Setup</a>

The four tools mentioned above are included in a Python package and can be run under a shell terminal. It has been confirmed to work under OS X and Linux.



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

CCU_scoring -h
```

A manual page can be printed for specfic command using the `-h` flag. For example,

```bash
CCU_scoring score-nd -h
```

### Reference Validation Tool

To **validate the format of a reference** directory to make sure the files under the `reference_directory` have the correct format:

```bash
CCU_scoring validate-ref -ref <reference_directory>

# an example of reference validation
CCU_scoring validate-ref -ref test/reference/LDC_reference_sample
```

**Required Arguments**

 * `-ref`: reference directory


### Submission Validation Tool

**Norm Discovery/Emotion Detection/Valence Diarization/Arousal Diarization/Change Detection Validation**

To **validate the format of a nd/ed/vd/ad/cd submission** directory against a reference directory with a scoring index file:

```bash
CCU_scoring validate-nd -s <submission_directory> -ref <reference_directory> -i <scoring_index_file>
CCU_scoring validate-ed -s <submission_directory> -ref <reference_directory> -i <scoring_index_file>
CCU_scoring validate-vd -s <submission_directory> -ref <reference_directory> -i <scoring_index_file>
CCU_scoring validate-ad -s <submission_directory> -ref <reference_directory> -i <scoring_index_file>
CCU_scoring validate-cd -s <submission_directory> -ref <reference_directory> -i <scoring_index_file>

# an example of submission validation
CCU_scoring validate-nd \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220815_164235 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab
```

**Required Arguments**

 * `-s`: submission directory

 * `-ref`: reference directory

 * `-i`: file containing the file id of scoring datasets

**Norm Discovery Mapping Validation**

To **validate the format of a ndmap submission** directory against a reference directory with a scoring index file:

```bash
CCU_scoring validate-ndmap -s <submission_directory> -n <hidden_norm_list_file>

# an example of ndmap submission validation
CCU_scoring validate-ndmap \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/NDMAP/CCU_P1_TA1_NDMAP_NIST_mini-eval1_20220605_050236 \
-n test/hidden_norms.txt 
```

**Required Arguments**

 * `-s`: submission directory

 * `-n`: file containing the hidden norm


### Submission Scoring Tool

**Norm Discovery Scoring**

To **score a nd submission** directory against a reference directory with a scoring index file:

```bash
CCU_scoring score-nd -s <norm_submission_directory> -ref <reference_directory> -i <scoring_index_file>

# an example of norm scoring
CCU_scoring score-nd \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220815_164235 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab
```

**Norm Discovery Mapping Scoring**

To **score a ndmap submission** directory against a reference directory with a scoring index file:

```bash
CCU_scoring score-nd -s <norm_submission_directory> -m <norm_mapping_submission_directory> -ref <reference_directory> -i <scoring_index_file>

# an example of ndmap scoring
CCU_scoring score-nd \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220531_050236 \
-m test/pass_submissions/pass_submissions_LDC_reference_sample/NDMAP/CCU_P1_TA1_NDMAP_NIST_mini-eval1_20220605_050236 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab
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

**Emotion Detection Scoring**

To **score a ed submission** directory against a reference directory with a scoring index file:

```bash
CCU_scoring score-ed -s <norm_submission_directory> -m <norm_mapping_submission_directory> -ref <reference_directory> -i <scoring_index_file>

# an example of ed scoring
CCU_scoring score-ed \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/ED/CCU_P1_TA1_ED_NIST_mini-eval1_20220531_050236 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab
```
**Required Arguments**

 * `-s`: emotion submission directory

 * `-ref`: reference directory

 * `-i`: file containing the file id of scoring datasets

**Optional Arguments**

 * `-e`: file containing the emotion to filter emotion from scoring

 * `-t`: comma separated list of IoU thresholds

 * `-o`: output directory containing the score and alignment file

**Valence/Arousal Detection Scoring**

To **score a vd/ad submission** directory against a reference directory with a scoring index file:

```bash
CCU_scoring score-vd -s <norm_submission_directory> -m <norm_mapping_submission_directory> -ref <reference_directory> -i <scoring_index_file>
CCU_scoring score-ad -s <norm_submission_directory> -m <norm_mapping_submission_directory> -ref <reference_directory> -i <scoring_index_file>

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
**Required Arguments**

 * `-s`: emotion submission directory

 * `-ref`: reference directory

 * `-i`: file containing the file id of scoring datasets

**Optional Arguments**

 * `-o`: output directory containing the score and diarization file

**Change Detection Scoring**

To **score a cd submission** directory against a reference directory with a scoring index file:

```bash
CCU_scoring score-cd -s <norm_submission_directory> -m <norm_mapping_submission_directory> -ref <reference_directory> -i <scoring_index_file>

# an example of cd scoring
CCU_scoring score-cd \
-s test/pass_submissions/pass_submissions_LDC_reference_sample/CD/CCU_P1_TA1_CD_NIST_mini-eval1_20220531_050236 \
-ref test/reference/LDC_reference_sample \
-i test/reference/LDC_reference_sample/index_files/LC1-SimulatedMiniEvalP1.20220909.scoring.index.tab 
```
**Required Arguments**

 * `-s`: emotion submission directory

 * `-ref`: reference directory

 * `-i`: file containing the file id of scoring datasets

**Optional Arguments**

 * `-e`: comma separated list of delta CP text thresholds

 * `-m`: comma separated list of delta CP time thresholds

 * `-o`: output directory containing the score and alignment file

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