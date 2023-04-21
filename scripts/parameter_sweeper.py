#! /usr/bin/env python
"""
A python script to run a set of scooring varitions and plot them.

"""
import os, glob
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import argparse
import pathlib
import matplotlib.pyplot as plt
import json
import sys
from PIL import Image

def add_scorer_file(file, srid, tab):
    t = pd.read_csv(file, sep = "\t")
    t['srid'] = srid

    if (tab is None):
        tab = t
    else:
        tab = pd.concat([tab, t])
    return(tab)

def load_scoring(dir, id, lab, df_jobs, df_sr, df_sbc, df_sa, df_sp):
    ### Jobs table
    if (df_jobs is None):
        df_jobs = pd.DataFrame([], columns = ['jobid'])
    if (len(df_jobs[df_jobs.jobid == id].jobid) == 0):
        df_jobs = pd.concat([ df_jobs, pd.DataFrame([id], columns = ['jobid']) ])

    ### Add a scoring run
    if (df_sr is None):
        df_sr = pd.DataFrame([], columns = ['jobid', 'srid', 'label'])
    srid = len(df_sr.srid)
    df_sr = pd.concat([ df_sr, pd.DataFrame([ [id, srid, lab] ], columns = ['jobid', 'srid', 'label']) ])

    df_sbc = add_scorer_file(os.path.join(dir, "scores_by_class.tab"), srid, df_sbc)
    df_sa = add_scorer_file(os.path.join(dir, "scores_aggregated.tab"), srid, df_sa)
    df_sp = add_scorer_file(os.path.join(dir, "scoring_parameters.tab"), srid, df_sp)
    return(df_jobs, df_sr, df_sbc, df_sa, df_sp)

def run_score(code_base, command, workdir, param1):
    print("Entering run_score()")

    ### make the workdir
    if (not os.path.isdir(workdir)):
        os.mkdir(workdir)
        print(f"   mkdir {workdir}")

    ### Initialize the scores DFs - To be None - The reader handles the initial build
    df_jobs, df_sr, df_sbc, df_sa, df_sp = [None, None, None, None, None]

    for level in param1['levels']:
        lev_out = os.path.join(workdir, level['name'])
        run = False
        if (not os.path.isdir(lev_out)):
            run = True

        ######  Scorer execution ####
        if (not run):
            print(f"   Beginning level: {level['name']} args: /{level['args']}/ output: {lev_out}")
        else:
            os.mkdir(lev_out)
            shfile, retfile = [f"{lev_out}.sh", f"{lev_out}.ret"]
            print(f"   Beginning level: {level['name']} args: /{level['args']}/ outputr: {lev_out}  Starting scorer: {shfile}")
            if (code_base == "CCU_scoring"):
                com = code_base + " " + command + " " + level['args'] + " -o " + lev_out
            else:
                com = "( cd " + code_base + " ; python -m CCU_validation_scoring.cli " + command + " " + level['args'] + " -o " + str(pathlib.Path(lev_out).resolve()) + ")"
            print(com + f" 1> {lev_out}.stdout.txt 1> {lev_out}.stderr.txt", file=open(shfile, 'w'))
            ret = os.system(f"sh {shfile}")
            print(ret, file=open(retfile, 'w'))
            if (ret != 0):
                print(f"Error: Scorer returned /{ret}/.  Aborting")

        ### Load tqbles
        df_jobs, df_sr, df_sbc, df_sa, df_sp = load_scoring(lev_out, "sweep", level['name'], df_jobs, df_sr, df_sbc, df_sa, df_sp)        
    

    # print(df_jobs)
    # print(df_sr)
    # print(df_sbc)
    # print(df_sa)
    # print(df_sp)    
    
    return(df_jobs, df_sr, df_sbc, df_sa, df_sp)


def plot_measures_agg(df_jobs, df_sr, df_sbc, df_sa, df_sp, measures, outdir):
    images = []
    mean_measures = [ "mean_" + m for m in measures ]
    ##############################
    ### metrics aggregated by 'all'
    me = pd.merge(df_sa, df_sr, on='srid')
    fig, ax = plt.subplots(1, len(mean_measures), figsize=(9, 3), sharey=True)
    fig.subplots_adjust(hspace=0.1, top=0.8, bottom=0.2)
    i = 0
    for mea in mean_measures:
        x = me[(me.metric.isin([mea]) & me.genre.isin(['all']))]
        ax[i].plot(x.label, x.value, label='all')
        ax[i].title.set_text(mea)
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(45)            
        ax[i].legend(fontsize="6", loc ="upper right")
        i = i + 1
    file = os.path.join(outdir, "plots.agg.png")
    images.append(file)
    print(f"Producing {file}")
    plt.savefig(file, dpi=300)
    plt.close()
    
    ##############################
    ### metrics aggregated by type
    fig, ax = plt.subplots(1, len(mean_measures), figsize=(9, 3), sharey=True)
    fig.subplots_adjust(hspace=0.1, top=0.8, bottom=0.2)
    i = 0
    for mea in mean_measures:
        for type in ['audio', 'video', 'text']:
            x = me[(me.metric.isin([mea]) & me.genre.isin([type]))]
            ax[i].plot(x.label, x.value, label=type)
            ax[i].title.set_text(mea)
            
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(45)
        ax[i].legend(fontsize="6", loc ="upper right")
        i = i + 1
    file = os.path.join(outdir, "plots.agg.type.png")
    images.append(file)
    print(f"Producing {file}")
    plt.savefig(file, dpi=300)
    plt.close()

    ##############################
    ### metrics by class
    me = pd.merge(df_sbc, df_sr, on='srid')
    me = me[me.metric.isin(measures)]
    me.value = [ float(x) for x in me.value]
    fig, ax = plt.subplots(1, len(measures), figsize=(9, 3), sharey=True)
    fig.subplots_adjust(hspace=0.1, top=0.8, bottom=0.2)
    i = 0
    for mea in measures:
        for clas in sorted(set(me['class'])):
            x = me[(me.metric.isin([mea]) & me.genre.isin(['all']) & me['class'].isin([clas]))]
            ax[i].plot(x.label, x.value, label=clas)
            ax[i].title.set_text(mea)
            
        for tick in ax[i].get_xticklabels():
            tick.set_rotation(45)
        ax[i].legend(fontsize="6", loc ="upper right")
        i = i + 1
    file = os.path.join(outdir, "plots.class.png")
    images.append(file)
    print(f"Producing {file}")
    plt.savefig(file, dpi=300)
    plt.close()

    ##############################
    ### metrics by class and type
    me = pd.merge(df_sbc, df_sr, on='srid')
    me = me[me.metric.isin(measures)]
    me.value = [ float(x) for x in me.value]
    for type in ['audio', 'video', 'text']:
        fig, axs = plt.subplots(1, len(measures), figsize=(9, 3), sharey=True)
        fig.subplots_adjust(hspace=0.1, top=0.8)
        i = 0
        for mea in measures:
            for clas in sorted(set(me['class'])):
                x = me[(me.metric.isin([mea]) & me.genre.isin([type]) & me['class'].isin([clas]))]
                axs[i].plot(x.label, x.value, label=clas)
                axs[i].title.set_text(mea)
            
            for tick in axs[i].get_xticklabels():
                tick.set_rotation(45)
            axs[i].legend()
            i = i + 1
        plt.suptitle(type + " Data", size=16)
        file = os.path.join(outdir, "plots.class." + type + ".png")
        images.append(file)
        print(f"Producing {file}")
        plt.savefig(file, dpi=300)
        plt.close()

    ##############################
    ### PRCurves
    me = pd.merge(df_sbc, df_sr, on='srid')
    me = me[me.metric.isin(['PRCurve_json'])]

    types = sorted(set(me['genre']))
    #print(sorted(set(me['class'])))
    for clas in sorted(set(me['class'])):
        fig, ax = plt.subplots(1, len(types), figsize=(9, 3), sharey=True)
        fig.subplots_adjust(hspace=0.1, top=0.8)
        i = 0
        for type in types:
            ax[i].set(xlim=(0, 1), xticks=np.arange(0, 1.01, 0.2),
                      ylim=(0, 1), yticks=np.arange(0, 1.01, 0.1))
            if (i == 0):
                ax[i].set_ylabel('Precision')
            ax[i].set_xlabel('Recall')
            ax[i].set_title(f"{type}")
            x = me[(me.genre.isin([type]) & me['class'].isin([clas]))]
            for index, row in x.iterrows():
                prc = json.loads(row['value'])
                ax[i].plot(prc['recall'], prc['precision'], linewidth=1.0, label=row['label'])
            [ tick.set_size(8) for tick in ax[i].get_xticklabels() ]
            [ tick.set_size(8) for tick in ax[i].get_yticklabels() ]
            ax[i].legend(fontsize="6", loc ="upper right")
            i = i + 1
        plt.suptitle(str(clas) + " norm", size=16)        
        file = os.path.join(outdir, "plots.prc." + str(clas) + ".png")
        images.append(file)
        print(f"Producing {file}")
        plt.savefig(file, dpi=300)
        plt.close()

    imgs = [Image.open(x) for x in images]
    widths, heights = zip(*(i.size for i in imgs))

    total_width = max(widths)
    max_height = sum(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    y_offset = 0
    fCor im in imgs:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    new_im.save('test.jpg')
        
def main():
    parser = argparse.ArgumentParser(description='Generate a random norm/emotion submission')
    
    parser.add_argument('-C', '--code_base', type=str, required=True, default='CCU_scoring', help = 'Either the Git Repo dir or "CCU_scoring" (defualt)')
    parser.add_argument('-c', '--base_command', type=str, required=True, help = 'This is the base scoring command without the executeable name')
    parser.add_argument('-t', '--task', choices=['norm', 'emotion'], required=True, help = 'norm, emotion')
    parser.add_argument('-w', '--workdir', type=str, required=True, help = 'The working director`y to store results')
    parser.add_argument(      '--param1_file', type=str, required=True, help = 'The param def JSON')
    parser.add_argument('-m', '--measures', type=str, default = "average_precision f1_at_MinLLR scaled_f1_at_MinLLR", help = 'The statistics to plot - these are named for class metrics')

    
     
    args = parser.parse_args()
    with open(args.param1_file,"r") as f: param1_txt = f.read()
    param1 = json.loads(param1_txt)

    ### Build the factor List
    param1['factors'] = { }
    for level in param1['levels']:
        if ('factors' in level):
            for fact, lev in level['factors'].items():
                if (fact not in param1['factors']):
                    param1['factors'][fact] = []
                if (lev not in param1['factors'][fact]):
                    param1['factors'][fact].append(lev)    
    
    #print(json.dumps(param1, indent=4), file=open(args.param1_file, 'w'))
    df_jobs, df_sr, df_sbc, df_sa, df_sp = run_score(args.code_base, args.base_command, args.workdir, param1)
    measures = args.measures.split(" ")
    plot_measures_agg(df_jobs, df_sr, df_sbc, df_sa, df_sp, measures, args.workdir)


if __name__ == '__main__':
	main()
