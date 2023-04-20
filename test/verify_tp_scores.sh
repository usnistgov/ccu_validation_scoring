#!/bin/sh

dir=$1
nfp=`cat $dir/instance_alignment.tab | awk 'BEGIN{s=0} {if ($3 == "unmapped" && $4 == "{}"){s=s+1}} END {print s}'`
sfp=`cat $dir/instance_alignment.tab | grep pct_temp_fp= | perl -pe 's/.*pct_temp_fp=//; s/,.*//' | awk '{s=s+$1}END{print s}'`
stp=`cat $dir/instance_alignment.tab | grep pct_temp_tp= | perl -pe 's/.*pct_temp_tp=//; s/,.*//' | awk '{s=s+$1}END{print s}'`

cl_sfp=`grep sum_scaled_fp_at_MinLLR $dir/scores_by_class.tab | grep all | awk '{s=s+$4}END{print s}'`
#echo SFP $nfp + $sfp = `echo $nfp $sfp | awk '{print $1 + $2}'`

tst=`echo $sfp $cl_sfp | awk '{d = ($1-$2); if (d < 0) {d = d * (-1)} if (d > 0.5) {print "FAIL "d} else {print "PASS"} }'`
if [ ! "$tst" = "PASS" -o "$2" = 'report' ] ; then
    echo "-------------"
    echo $dir
    grep sum_scaled_fp_at_MinLLR $dir/scores_aggregated.tab | grep all
    grep sum_scaled_fp_at_MinLLR $dir/scores_by_class.tab | grep all
    classes=`grep sum_scaled_fp_at_MinLLR $dir/scores_by_class.tab | grep all | awk '{print $1}'`
    echo SFP $sfp - class $cl_sfp $tst $dir

    for cl in $classes ; do
	echo $cl `cat $dir/instance_alignment.tab |egrep "^$cl" | grep pct_temp_fp= | perl -pe 's/.*pct_temp_fp=//; s/[,\}].*//' | awk '{s=s+$1}END{print s}'`
    done
    echo
    grep sum_scaled_tp_at_MinLLR $dir/scores_aggregated.tab | grep all
    grep sum_scaled_tp_at_MinLLR $dir/scores_by_class.tab | grep all
    cl_stp=`grep sum_scaled_tp_at_MinLLR $dir/scores_by_class.tab | grep all | awk '{s=s+$4}END{print s}'`
    echo STP $stp - class $cl_stp
    
    echo 
    cat $dir/instance_alignment.tab | head
fi



### Check all
### for f in `egrep "(ED|ND)" ../all_scorable_minieval_list.txt | xargs -I % sh maint_submission.sh -j % -o list | grep score.sh | grep _gdr | sed 's:/score.sh::'` ; do echo $f ; sh ../scorers/ccu_validation_scoring/test/verify_tp_scores.sh $f | head -20 ; done | grep FAI
