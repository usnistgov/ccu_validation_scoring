#!/bin/sh

dir=$1
echo "-------------"
echo $dir
nfp=`cat $dir/instance_alignment.tab | awk 'BEGIN{s=0} {if ($3 == "unmapped" && $4 == "{}"){s=s+1}} END {print s}'`
sfp=`cat $dir/instance_alignment.tab | grep pct_temp_fp= | perl -pe 's/.*pct_temp_fp=//; s/,.*//' | awk '{s=s+$1}END{print s}'`
stp=`cat $dir/instance_alignment.tab | grep pct_temp_tp= | perl -pe 's/.*pct_temp_tp=//; s/,.*//' | awk '{s=s+$1}END{print s}'`

grep sum_scaled_fp_at_MinLLR $dir/scores_aggregated.tab | grep all
grep sum_scaled_fp_at_MinLLR $dir/scores_by_class.tab | grep all
cl_sfp=`grep sum_scaled_fp_at_MinLLR $dir/scores_by_class.tab | grep all | awk '{s=s+$4}END{print s}'`
#echo SFP $nfp + $sfp = `echo $nfp $sfp | awk '{print $1 + $2}'`
echo SFP $sfp - class $cl_sfp
echo
grep sum_scaled_tp_at_MinLLR $dir/scores_aggregated.tab | grep all
grep sum_scaled_tp_at_MinLLR $dir/scores_by_class.tab | grep all
cl_stp=`grep sum_scaled_tp_at_MinLLR $dir/scores_by_class.tab | grep all | awk '{s=s+$4}END{print s}'`
echo STP $stp - class $cl_stp

echo 
cat $dir/instance_alignment.tab
#grep sum_scaled_fp_at_MinLLR $dir/scores_by_class.tab | grep all


