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

def compute_stats(df, file):

    df['duration'] = df['end'] - df['start']
    stats = pd.DataFrame(columns = ['class', 'genre', 'file', 'file_count', 'instance_count', 'unit', 'duration', 'mean', 'stdev', 'min', 'p25', 'p50', 'p75', 'max'])
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
                'file': file,
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
    parser.add_argument("-vR", "--merge_ref_label", type=str, choices=['class', 'class-status'], required=False, help="choose class or class-status to define how to handle the adhere/violate labels for the reference norm instances merging.")
    parser.add_argument('-t', '--task', choices=['norms', 'emotions'], required=True, help = 'norms, emotions')
    parser.add_argument('-i', '--scoring-index-file', type=str, required=True, help='use to filter file from scoring (REF)')
    parser.add_argument('-o', '--output-file', type=str, required=True, help='file where the statistics will be output')    

    args = parser.parse_args()
    
    try:
        scoring_index = pd.read_csv(args.scoring_index_file, usecols = ['file_id'], sep = "\t")
    except Exception as e:
        print('ERROR:GENERATING:{} is not a valid scoring index file'.format(args.scoring_index_file))
        exit(1)

    if args.merge_ref_text_gap:
        merge_ref_text_gap = int(args.merge_ref_text_gap)
    else:
        merge_ref_text_gap = None
    
    if args.merge_ref_time_gap:
        merge_ref_time_gap = float(args.merge_ref_time_gap)
    else:
        merge_ref_time_gap = None

    stats_all = pd.DataFrame()

    for i in range(scoring_index.shape[0]):
        file = scoring_index.iloc[i].to_frame().T
        file.reset_index(drop = True, inplace = True)

        if args.task == "norms":
            ref = preprocess_reference_dir(args.ref_dir, file, args.task, merge_ref_text_gap, merge_ref_time_gap, args.merge_ref_label, False, None)
        else:
            ref = preprocess_reference_dir(args.ref_dir, file, args.task, merge_ref_text_gap, merge_ref_time_gap, None, False, None)

        stats = compute_stats(ref, file["file_id"].values[0])
        stats_all = pd.concat([stats_all, stats])
    
    stats_all.to_csv("{}_PER_FILE.txt".format(args.output_file), sep='\t', index=False, float_format='%.2f')

if __name__ == "__main__":
    main()
