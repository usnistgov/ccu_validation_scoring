
c="python -m CCU_validation_scoring.cli score-nd -ref /Users/jon/Programs/CCU/ccu_validation_scoring_BR_one2many/test/reference/AlignFile_tests -s /Users/jon/Programs/CCU/ccu_validation_scoring_BR_one2many/test/pass_submissions/pass_submissions_AlignFile_tests/ND/system1 -i /Users/jon/Programs/CCU/ccu_validation_scoring_BR_one2many/test/reference/AlignFile_tests/index_files/AlignFile_tests.scoring_input.index.tab -o /Users/jon/Programs/CCU/ccu_validation_scoring_BR_one2many/test/test_output  -n /Users/jon/Programs/CCU/ccu_validation_scoring_BR_one2many/test/known_norms_AlignFile_tests_102.txt -aC 15 -xC 150 -t intersection:gt:0 -q"

echo "OneRef:OneHyp - No merge"
echo "This is similar to Joe's example of needing to split the system accross three reference instances"
echo ""
$c > /dev/null
cat /Users/jon/Programs/CCU/ccu_validation_scoring_BR_one2many/test/test_output/instance_alignment.tab  | ~/Tools/MIGTools/tabby/tabby.py -s "\t" --no_show

echo ""
echo "ManyRef:OneHyp - No merge"
echo $c --align_hacks "ManyRef:OneHyp" 
$c --align_hacks "ManyRef:OneHyp" > /dev/null
cat /Users/jon/Programs/CCU/ccu_validation_scoring_BR_one2many/test/test_output/instance_alignment.tab  | ~/Tools/MIGTools/tabby/tabby.py -s "\t" --no_show

echo ""
echo "ManyRef:OneHyp - RefMerge=30, SysMerge=30"
$c --align_hacks "ManyRef:OneHyp" -aS 30 -xS 300 -lS max_llr -vS class -aR 30 -xR 300 -vR class > /dev/null
cat /Users/jon/Programs/CCU/ccu_validation_scoring_BR_one2many/test/test_output/instance_alignment.tab  | ~/Tools/MIGTools/tabby/tabby.py -s "\t" --no_show


echo ""
echo "ManyRef:OneHyp - RefMerge=30, SysMerge=NONE"
$c --align_hacks "ManyRef:OneHyp" -aR 30 -xR 300 -vR class > /dev/null
cat /Users/jon/Programs/CCU/ccu_validation_scoring_BR_one2many/test/test_output/instance_alignment.tab  | ~/Tools/MIGTools/tabby/tabby.py -s "\t" --no_show


#--align_hacks ManyRef:OneHyp | m
