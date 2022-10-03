#!/bin/sh

ref=SmoothCurve.ref.json
sys=SmoothCurve.sys.json

echo "Make the segments"
echo "file_id	segment_id	start	end" > docs/segments.tab
grep '"Closing"' SmoothCurve.ref.json | sed 's/.mp4//g'   | sed 's/"/ /g' | awk 'BEGIN {n=1}; {print $10"\t"$10"_"n"\t"$12"\t"$14-50; print $10"\t"$10"_"(n+1)"\t"$14-50"\t"$14; n+=2}' >> docs/segments.tab
echo 'VIRAT_S_000000	VIRAT_S_000000_17	800	850' >> docs/segments.tab
echo 'VIRAT_S_000000	VIRAT_S_000000_18	850	900' >> docs/segments.tab
echo 'VIRAT_S_000000	VIRAT_S_000000_19	900	950' >> docs/segments.tab
echo 'VIRAT_S_000000	VIRAT_S_000000_20	950	1000' >> docs/segments.tab
echo 'VIRAT_S_000000	VIRAT_S_000000_21	1000	1050' >> docs/segments.tab
echo 'VIRAT_S_000000	VIRAT_S_000000_22	1050	1100' >> docs/segments.tab
echo 'VIRAT_S_000000	VIRAT_S_000000_23	1100	1150' >> docs/segments.tab
echo 'VIRAT_S_000000	VIRAT_S_000000_24	1150	1200' >> docs/segments.tab

echo "Make the reference"
echo "user_id	file_id	segment_id	norm	status" > data/norms.tab
grep '"Closing"' SmoothCurve.ref.json | sed 's/.mp4//g'   | sed 's/"/ /g' | awk '{print "1001\t"$10"\t"$10"_"(NR)"\tclosing\tadhere"}' | sed 's/C/c/'  >> data/norms.tab

echo "Make the index"
(echo "file_id" ; echo VIRAT_S_000000) > index_files/ActEV-SmoothCurve.system_input.index.tab
(echo "file_id" ; echo VIRAT_S_000000) > index_files/ActEV-SmoothCurve.scoring.index.tab


echo "Make the system"
sdir=../../pass_submissions/pass_submissions_ActEV-SmoothCurve/ND/CCU_P1_TA1_ND_NISTACTEV_mini-eval1_20220930_050236/
mkdir -p $sdir

echo "the system index"
echo 'file_id	is_processed	message	file_path' > $sdir/system_output.index.tab
echo 'VIRAT_S_000000	True	foo	VIRAT_S_000000.tab' >> $sdir/system_output.index.tab

echo "The system output"
echo 'file_id	norm	start	end	status	llr' > $sdir/VIRAT_S_000000.tab
grep '"Closing"' SmoothCurve.sys.json | sed 's/.mp4//g'    | grep CD | sed 's/"/ /g' | perl -pe 's/:\s(\d)/:$1/' | awk '{print $18"\tclosing\t"$20"\t"$22"\tadhere\t"$9}' | sed 's/[:,]//g' >>  $sdir/VIRAT_S_000000.tab
grep '"Closing"' SmoothCurve.sys.json | sed 's/.mp4//g'    | grep FA | sed 's/"/ /g' | perl -pe 's/:\s(\d)/:$1/' | awk '{print $18"\tclosing\t"$20"\t"$22"\tviolate\t"$9}' | sed 's/[:,]//g' >>  $sdir/VIRAT_S_000000.tab
###M010005NJ       001     1       4       adhere  0.6
