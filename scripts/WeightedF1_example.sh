mkdir -p outdir
python -m CCU_validation_scoring.cli score-nd \
       -ref test/reference/WeightedF1 \
       -s test/pass_submissions/pass_submissions_WeightedF1/ND/system1 \
       -i test/reference/WeightedF1/index_files/WeightedF1.scoring_input.index.tab \
       -o outdir \
       -n test/known_norms_WeightedF1.txt \
       -aR 30 -xR 300 -vR class -t intersection:gt:0 -aC 15 -xC 150
       
       