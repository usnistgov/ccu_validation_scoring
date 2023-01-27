#! /usr/bin/env python
"""
A python script which reads the LDC CCU reference directory and
writes out basic summary statistics. Currently, it only works
for norms and emotions.
"""
import sys
import argparse
import pandas as pd
from CCU_validation_scoring.preprocess_reference import *

def compute_stats(df):

    df['duration'] = df['end'] - df['start']
    stats = pd.DataFrame(columns = ['class', 'genre', 'file_count', 'instance_count', 'unit', 'duration', 'mean', 'stdev', 'min', 'p25', 'p50', 'p75', 'max'])
    for genre in df['type'].unique():
        for clss in df[df['type'] == genre]['Class'].unique():
            duration_df = df[(df['type'] == genre) & (df['Class'] == clss)]
            info = duration_df['duration'].describe()
            if (genre == "text"):
                unit = "character"
            else:
                unit = "second"
            stats = pd.concat([stats, pd.DataFrame.from_records([{
                'class': clss,
                'genre': genre,
                'file_count': len(duration_df['file_id'].unique()),
                'instance_count': len(duration_df),
                'unit': unit,
                'duration': sum(duration_df['duration']),
                'mean': info[1],
                'stdev': info[2],
                'min': info[3],
                'p25': info[4],
                'p50': info[5],
                'p75': info[6],
                'max': info[7]}])])
    return(stats)                
    
def main():
    parser = argparse.ArgumentParser(description="Compute some statistics on the reference data.")
    parser.add_argument('-r', '--ref-dir', type=str, required=True, help="path to the reference directory")
    parser.add_argument("-xR", "--merge_ref_text_gap", type=str, required=False, help="merge reference text gap character")
    parser.add_argument("-aR", "--merge_ref_time_gap", type=str, required=False, help="merge reference time gap second")
    parser.add_argument('-t', '--task', choices=['norms', 'emotions'], required=True, help = 'norms, emotions')
    parser.add_argument('-i', '--scoring-index-file', type=str, required=True, help='use to filter file from scoring (REF)')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='file where the statistics will be output')    

    args = parser.parse_args()
    ref_dir = args.ref_dir
    task = args.task
    scoring_index_file = args.scoring_index_file
    output_file = args.output_file
    
    try:
        scoring_index = pd.read_csv(scoring_index_file, usecols = ['file_id'], sep = "\t")
    except Exception as e:
        print('ERROR:GENERATING:{} is not a valid scoring index file'.format(scoring_index_file))
        exit(1)

    if args.merge_ref_text_gap:
        merge_ref_text_gap = int(args.merge_ref_text_gap)
    else:
        merge_ref_text_gap = None
    
    if args.merge_ref_time_gap:
        merge_ref_time_gap = float(args.merge_ref_time_gap)
    else:
        merge_ref_time_gap = None

    ref = preprocess_reference_dir(ref_dir, scoring_index, task, merge_ref_text_gap, merge_ref_time_gap)

    stats = compute_stats(ref)
    stats.to_csv(output_file, sep='\t', index=False, float_format='%.2f')

if __name__ == "__main__":
    main()
