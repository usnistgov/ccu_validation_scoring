# Installation

```bash
cd /path/to/CCU_validation_scoring
python3 -m pip install -e ./
```

# Usage

```bash
# General Help
CCU_scoring -h

# Command Specific Help
CCU_scoring score-nd -h

# Validate Norm Detection Task
CCU_scoring validate-nd -s test/submission/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220815_164235 -ref test/reference/LDC_reference_sample

# Score Norm Detection Task
CCU_scoring score-nd -s test/submission/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220815_164235 -ref test/reference/LDC_reference_sample

# Score Norm Detection Task with a mapping file
CCU_scoring score-nd -s test/submission/ND/CCU_P1_TA1_ND_NIST_mini-eval1_20220531_050236 -ref test/reference/LDC_reference_sample -m test/submission/NDMAP/CCU_P1_TA1_NDMAP_NIST_mini-eval1_20220605_050236

# Validate Emotion Detection Task
CCU_scoring validate-ed -s test/submission/ED/CCU_P1_TA1_ED_NIST_mini-eval1_20220531_050236 -ref test/reference/LDC_reference_sample

# Score Emotion Detection Task
CCU_scoring score-ed -s test/submission/ED/CCU_P1_TA1_ED_NIST_mini-eval1_20220531_050236 -ref test/reference/LDC_reference_sample 

# Validate Valence Detection Task
CCU_scoring validate-vd -s test/submission/VD/CCU_P1_TA1_VD_NIST_mini-eval1_20220531_050236 -ref test/reference/LDC_reference_sample 

# Score Valence Detection Task
CCU_scoring score-vd -s test/submission/VD/CCU_P1_TA1_VD_NIST_mini-eval1_20220531_050236 -ref test/reference/LDC_reference_sample

# Validate Arousal Detection Task
CCU_scoring validate-ad -s test/submission/AD/CCU_P1_TA1_AD_NIST_mini-eval1_20220531_050236 -ref test/reference/LDC_reference_sample

# Score Arousal Detection Task
CCU_scoring score-ad -s test/submission/AD/CCU_P1_TA1_AD_NIST_mini-eval1_20220531_050236 -ref test/reference/LDC_reference_sample 

# Score Change Detection Task
CCU_scoring score-cd -s test/submission/CD/CCU_P1_TA1_CD_NIST_mini-eval1_20220531_050236 -ref test/reference/LDC_reference_sample 

```

# Authors

Jennifer Yu (yan.yu@nist.gov)
Lukas Diduch (lukas.diduch@nist.gov)

# Licensing Statement

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