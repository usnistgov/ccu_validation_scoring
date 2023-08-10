#! /usr/bin/env python
"""
A python script to run a set of scooring varitions and plot them.

"""
import os, glob
import math
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
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

def add_scorer_file(file, srid, tab):
    t = pd.read_csv(file, sep = "\t")
    t['srid'] = srid

    if (tab is None):
        tab = t
    else:
        tab = pd.concat([tab, t])
    return(tab)

def load_scoring(dir, id, lab, factors, df_jobs, df_sr, df_sbc, df_sa, df_sp, df_fact, df_sr_fact):
    ### Jobs table
    if (df_jobs is None):
        df_jobs = pd.DataFrame([], columns = ['jobid'])
    if (len(df_jobs[df_jobs.jobid == id].jobid) == 0):
        df_jobs = pd.concat([ df_jobs, pd.DataFrame([id], columns = ['jobid']) ])

    ### Factors may not always be present. (x=>1, y>=1) and (z=>1) are legal
    if (df_fact is None):
        df_fact = pd.DataFrame(sorted(factors.keys()), columns = ['factor'])
    else: ### check and add new factors
        new_f = factors.keys() - df_fact['factor'].to_list()
        if (len(new_f) > 0):
            df_fact = pd.concat([ df_fact,  pd.DataFrame(sorted(new_f), columns = ['factor']) ])
    
    ### Add a scoring run
    if (df_sr is None):
        df_sr = pd.DataFrame([], columns = ['jobid', 'srid', 'label', 'factor_ensemble', 'num_variates'])
    srid = len(df_sr.srid)

    df_sr = pd.concat([ df_sr, pd.DataFrame([ [id, srid, lab, ' '.join(sorted(factors.keys())), len(factors.keys())] ], 
                                            columns = ['jobid', 'srid', 'label', 'factor_ensemble', "num_variates"]) ] )

    # add the sr_fact join data
    if (df_sr_fact is None):
        df_sr_fact = pd.DataFrame([], columns = ['srid', 'factor', 'value'])
    df_sr_fact = pd.concat([ df_sr_fact, pd.DataFrame([ [srid, fact, value] for fact, value in factors.items() ], 
                                                      columns = ['srid', 'factor', 'value']) ] )

    df_sbc = add_scorer_file(os.path.join(dir, "scores_by_class.tab"), srid, df_sbc)
    df_sa = add_scorer_file(os.path.join(dir, "scores_aggregated.tab"), srid, df_sa)
    df_sp = add_scorer_file(os.path.join(dir, "scoring_parameters.tab"), srid, df_sp)
    return(df_jobs, df_sr, df_sbc, df_sa, df_sp, df_fact, df_sr_fact)

def run_score(code_base, command, workdir, param1):
    print("Entering run_score()")

    ### make the workdir
    if (not os.path.isdir(workdir)):
        os.mkdir(workdir)
        print(f"   mkdir {workdir}")

    ### Initialize the scores DFs - To be None - The reader handles the initial build
    df_jobs, df_sr, df_sbc, df_sa, df_sp, df_fact, df_sr_fact = [None, None, None, None, None, None, None]

    for scoring_run in param1['scoring_runs']:
        lev_out = os.path.join(workdir, scoring_run['name'])
        run = False
        if (not os.path.isdir(lev_out)):
            run = True 
        if (not os.path.isfile(os.path.join(lev_out, 'scoring_parameters.tab'))):
            run = True        

        ######  Scorer execution ####
        if (not run):
            print(f"   Beginning scoring_run: {scoring_run['name']}")# args: /{scoring_run['args']}/ output: {lev_out}")
        else:
            if (not os.path.isdir(lev_out)):
                os.mkdir(lev_out)
            shfile, retfile = [f"{lev_out}.sh", f"{lev_out}.ret"]
            print(f"   Beginning scoring_run: {scoring_run['name']}")# args: /{scoring_run['args']}/ output: {lev_out}  Starting scorer: {shfile}")
            if (code_base == "CCU_scoring"):
                com = code_base + " " + command + " " + scoring_run['args'] + " -o " + lev_out
            else:
                com = "( cd " + code_base + " ; python -m CCU_validation_scoring.cli " + command + " " + scoring_run['args'] + " -o " + str(pathlib.Path(lev_out).resolve()) + ")"
            print(com + f" 1> {lev_out}.stdout.txt 2> {lev_out}.stderr.txt", file=open(shfile, 'w'))
            ret = os.system(f"sh {shfile}")
            if (ret != 0):
                print(f"Error: Scorer returned /{ret}/.  Aborting")
                exit(1)
            print(ret, file=open(retfile, 'w'))

        ### Load tqbles
        df_jobs, df_sr, df_sbc, df_sa, df_sp, df_fact, df_sr_fact = load_scoring(lev_out, "sweep", scoring_run['name'], scoring_run['factors'],
                                                                                 df_jobs, df_sr, df_sbc, df_sa, df_sp, df_fact, df_sr_fact)
    if (True):
        print("df_jobs")
        print(df_jobs.head())
        print("df_sr")
        print(df_sr.head())
        print("df_sbc")
        print(df_sbc.head())
        print("df_sa")
        print(df_sa.head())
        print("df_sp")
        print(df_sp.head())
        print("df_fact")
        print(df_fact.head())
        print("df_sr_fact")
        print(df_sr_fact.head())

    return(df_jobs, df_sr, df_sbc, df_sa, df_sp, df_fact, df_sr_fact)

def merge_images(images, file):
    imgs = [Image.open(x) for x in images]
    widths, heights = zip(*(i.size for i in imgs))
    
    total_width = max(widths)
    max_height = sum(heights)
    
    new_im = Image.new('RGB', (total_width, max_height))
    
    y_offset = 0
    for im in imgs:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
        
    print(f"Producing {file}")
    new_im.save(file)
    

def plot_measures_agg(df_jobs, df_sr, df_sbc, df_sa, df_sp, df_fact, df_sr_fact, measures, outdir):
    images = []
    mean_measures = [ "mean_" + m for m in measures ]

    for stat in [ 'agg', 'class' ]:
        if (stat == 'agg' ):
            me = pd.merge(df_sa,
                          pd.merge(df_sr, df_sr_fact.pivot(index='srid', columns='factor', values='value'), on='srid'),
                          on='srid')
            met = 'mean_' + 'average_precision'
        else:
            me = pd.merge(df_sbc,
                          pd.merge(df_sr, df_sr_fact.pivot(index='srid', columns='factor', values='value'), on='srid'),
                          on='srid')
            met = met
        print(me)
        for genre in [ 'all' ]:
            print(f"Building {stat} {genre}")
            # #############  Univariate Plots
            fe_list = list(set(me[me['num_variates'] == 1]['factor_ensemble']))
            print(fe_list)
            if len(fe_list) > 0:
                fig = make_subplots(rows=len(fe_list), cols=1, subplot_titles=fe_list)
                r = 1
                pdf  = me[(me['genre'] == genre) & (me['metric'] == met) & (me['num_variates'] == 1) ].copy()
                pdf.value = [ float(x) for x in pdf.value ]
                value_max = pdf.value.max() * 1.1
                for fe in fe_list: 
                    print(f"   Processing Univariant {fe}")
                    for lev in sorted(set(me[ (me['factor_ensemble'] == fe) ][fe])):
                        fig.add_trace(go.Box(x=pdf[pdf[fe] == lev].value), row=r, col=1) 
                        fig.data[-1].name = lev
                        #fig.update_xaxes(title_text="mAP", range=[0.0, value_max], row=r, col=1)
                    r = r + 1
                        
                fig.update_traces(showlegend = False)
                fig.update_layout(height=120 + 120*len(fe_list), width=800, title_text=f"Univariate Factors ({met}):")
                fig.write_image(f"/tmp/{stat}_univariate_{genre}.png") 
                
            # #############  Bivariate Plots   
            value_max = 0.3
            fe_list = list(set(me[me['num_variates'] == 2]['factor_ensemble']))
            for fe in fe_list:
                f1, f2 = fe.split(' ')
                print(f"   Processing Biivariant {f1}-{f2}")
                f1_levs = sorted(list(set(me[me['factor_ensemble'] == fe][f1])))
                f2_levs = sorted(list(set(me[me['factor_ensemble'] == fe][f2])), reverse=True)
                pdf = me[(me['genre'] == genre) & (me['metric'] == met) & (me['num_variates'] == 2) ].copy()
                pdf.value = [ float(x) for x in pdf.value ]
                value_max = pdf.value.max() * 1.1
                fig = make_subplots(rows=len(f1_levs), cols=1, subplot_titles=[ f1 + ":" + x for x in f1_levs])
                r = 1
                for f1_lev in f1_levs:
                    for f2_lev in f2_levs:
                        fig.add_trace(go.Box(x=pdf[(pdf[f1] == f1_lev) & (pdf[f2] == f2_lev) ].value), row=r, col=1) 
                        fig.data[-1].name = f2 + ":" + f2_lev
                        fig.update_xaxes(range=[0.0, value_max], row=r, col=1)
                    r = r + 1

                fig.update_traces(showlegend = False)
                h = 120 + (30 + 25*len(f2_levs))*len(f1_levs)
                #print(h)
                fig.update_layout(height=h, width=800, title_text=f"Bivariate Factors {met}: " + fe)
                fig.write_image(f"/tmp/{type}_bivariate_{fe.replace(' ','_')}_{genre}.png")






#    fig.add_trace(go.Box(x=udf[(udf['genre'] == 'all') & (udf['facet'] == 'all_male')].value),   row=1, col=1)
#    fig.data[-1].name = "all_male"

    # fig.add_trace(
    #     go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
    #     row=2, col=1
    # )
    


    # udf = None
    # for xvar in set(me[me['num_variates'] == 1]['factor_ensemble']):
    #     print(f"Processing Univariant {xvar}")
    #     xudf = me[(me['factor_ensemble'] == xvar) & (me['metric'] == 'mean_'+metric)].copy()
    #     xudf['facet'] = xudf[xvar]
    #     if (udf is None):
    #         udf = xudf
    #     else:
    #         print("concat")
    #         udf = pd.concat([udf, xudf])
    # print(udf)
    # fig=px.strip(udf[udf['genre'] == 'all'], color="facet", y="facet", x="value", facet_col='genre', facet_row="factor_ensemble")
    # fig.write_image("/tmp/test.png")





    

    # exit(0)
    
    # for genre in ['all', 'video', 'audio', 'text']:
    #     fig, axs = plt.subplots(1, len(mean_measures), figsize=(9, 3), sharey=True)
    #     if (len(mean_measures) == 1):
    #         axs = [axs]
    #     fig.subplots_adjust(hspace=0.1, top=0.8, bottom=0.2)
    #     i = 0
    #     factor_cols = sorted(factor_cols, reverse=True) 
    #     for mea in mean_measures:
    #         for f1 in sorted(set(me[factor_cols[0]])):
    #             x = me[(me.metric.isin([mea]) & me.genre.isin([genre]) & (me[factor_cols[0]] == f1)   )]
    #             if (len(factor_cols) == 1):
    #                 axs[i].plot([ "" for x in x.value ], x.value, label=f1, markersize=4, marker="o", linestyle="-")
    #             else:
    #                 axs[i].plot(x[factor_cols[1]], x.value, label=f1, markersize=4, marker="o", linestyle="-")                   
    #             axs[i].title.set_text(f"{genre} {mea}")
    #         for tick in axs[i].get_xticklabels():
    #             tick.set_rotation(45)
    #         axs[i].legend(fontsize="5")
    #         if (len(factor_cols) > 1):
    #             x_ori = me[(me.metric.isin([mea]) & me.genre.isin([genre]))][factor_cols + ["genre", "metric", "value"]]
    #             x = x_ori.sort_values(by='value', ascending=False)
    #             x['ordinal'] = range(0, len(x.index))
    #             print(f"Top factor scoring_run values {mea} {genre} {x.iloc[0].value} {x.iloc[0][factor_cols[0]]} {x.iloc[0][factor_cols[1]]} ")
    #             print(x.head())
            
    #         i = i + 1
    #     file = os.path.join(outdir, f"plots.agg.{genre}.png")
    #     images.append(file) 
    #     print(f"Producing {file}")
    #     plt.savefig(file, dpi=300)
    #     plt.close()
        
    # merge_images(images, os.path.join(outdir, f"plots.agg.png"))

    # ### Classes
    # images = []
    # me = pd.merge(df_sbc, df_sr, on='srid')
    # factor_cols = df_fact.factor
    # #measures = ['average_precision', 'f1_at_MinLLR', 'scaled_f1_at_MinLLR']
    # me = me[me.metric.isin(measures)]
    # me.value = [ float(x) for x in me.value]

    # for genre in ['all', 'video', 'audio', 'text']:
    #     for clas in sorted(set(me['class'])):
    #         fig, axs = plt.subplots(1, len(measures), figsize=(9, 3), sharey=True)
    #         if (len(mean_measures) == 1):
    #             axs = [axs]
    #         fig.subplots_adjust(hspace=0.1, top=0.8, bottom=0.2)
    #         i = 0
    #         for mea in measures:
    #             ymax = np.max(me[(me.metric.isin([mea]))].value) * 1.1
    #             for f1 in sorted(set(me[factor_cols[0]])):
    #             #for f1 in sorted(set(me[factor_cols[1]].to_list())):
    #                 x = me[(me.metric.isin([mea]) & me.genre.isin([genre]) & me['class'].isin([clas]) &
    #                         (me[factor_cols[0]] == f1))]
    #                 #print(x[ ['genre', 'class', factor_cols[0], factor_cols[1], 'metric', 'value']])
    #                 if (len(factor_cols) == 1):
    #                     axs[i].plot([ '' for x in x.value ], x.value, label=f1, markersize=4, marker="o", linestyle="-")
    #                 else:
    #                     axs[i].plot(x[factor_cols[1]], x.value, label=f1, markersize=4, marker="o", linestyle="-")
    #                 axs[i].title.set_text(f"{clas} {genre} {mea}")
    #                 axs[i].set(ylim=(0, ymax), yticks=np.arange(0, ymax, ymax/10))
                    
    #             for tick in axs[i].get_xticklabels():
    #                 tick.set_rotation(45)
    #             axs[i].legend()
    #             i = i + 1
    #         file = os.path.join(outdir, f"plots.class.{clas}.{genre}.png")
    #         images.append(file) 
    #         print(f"Producing {file}")
    #         plt.savefig(file, dpi=300)
    #         plt.close()
            
    # merge_images(images, os.path.join(outdir, f"plots.class.png"))






    
# def plot_measures_agg1(df_jobs, df_sr, df_sbc, df_sa, df_sp, measures, outdir):
#     images = []
#     mean_measures = [ "mean_" + m for m in measures ]
#     ##############################
#     ### metrics aggregated by 'all'
#     me = pd.merge(df_sa, df_sr, on='srid')
#     fig, ax = plt.subplots(1, len(mean_measures), figsize=(9, 3), sharey=True)
#     fig.subplots_adjust(hspace=0.1, top=0.8, bottom=0.2)
#     i = 0
#     for mea in mean_measures:
#         x = me[(me.metric.isin([mea]) & me.genre.isin(['all']))]
#         ax[i].plot(x.label, x.value, label='all')
#         ax[i].title.set_text(mea)
#         for tick in ax[i].get_xticklabels():
#             tick.set_rotation(45)            
#         ax[i].legend(fontsize="6", loc ="upper right")
#         i = i + 1
#     file = os.path.join(outdir, "plots.agg.png")
#     images.append(file)
#     print(f"Producing {file}")
#     plt.savefig(file, dpi=300)
#     plt.close()
    
#     ##############################
#     ### metrics aggregated by type
#     fig, ax = plt.subplots(1, len(mean_measures), figsize=(9, 3), sharey=True)
#     fig.subplots_adjust(hspace=0.1, top=0.8, bottom=0.2)
#     i = 0
#     for mea in mean_measures:
#         for type in ['audio', 'video', 'text']:
#             x = me[(me.metric.isin([mea]) & me.genre.isin([type]))]
#             ax[i].plot(x.label, x.value, label=type)
#             ax[i].title.set_text(mea)
            
#         for tick in ax[i].get_xticklabels():
#             tick.set_rotation(45)
#         ax[i].legend(fontsize="6", loc ="upper right")
#         i = i + 1
#     file = os.path.join(outdir, "plots.agg.type.png")
#     images.append(file)
#     print(f"Producing {file}")
#     plt.savefig(file, dpi=300)
#     plt.close()

#     ##############################
#     ### metrics by class
#     me = pd.merge(df_sbc, df_sr, on='srid')
#     me = me[me.metric.isin(measures)]
#     me.value = [ float(x) for x in me.value]
#     fig, ax = plt.subplots(1, len(measures), figsize=(9, 3), sharey=True)
#     fig.subplots_adjust(hspace=0.1, top=0.8, bottom=0.2)
#     i = 0
#     for mea in measures:
#         for clas in sorted(set(me['class'])):
#             x = me[(me.metric.isin([mea]) & me.genre.isin(['all']) & me['class'].isin([clas]))]
#             ax[i].plot(x.label, x.value, label=clas)
#             ax[i].title.set_text(mea)
            
#         for tick in ax[i].get_xticklabels():
#             tick.set_rotation(45)
#         ax[i].legend(fontsize="6", loc ="upper right")
#         i = i + 1
#     file = os.path.join(outdir, "plots.class.png")
#     images.append(file)
#     print(f"Producing {file}")
#     plt.savefig(file, dpi=300)
#     plt.close()

#     ##############################
#     ### metrics by class and type
#     me = pd.merge(df_sbc, df_sr, on='srid')
#     me = me[me.metric.isin(measures)]
#     me.value = [ float(x) for x in me.value]
#     for type in ['audio', 'video', 'text']:
#         fig, axs = plt.subplots(1, len(measures), figsize=(9, 3), sharey=True)
#         fig.subplots_adjust(hspace=0.1, top=0.8)
#         i = 0
#         for mea in measures:
#             for clas in sorted(set(me['class'])):
#                 x = me[(me.metric.isin([mea]) & me.genre.isin([type]) & me['class'].isin([clas]))]
#                 axs[i].plot(x.label, x.value, label=clas)
#                 axs[i].title.set_text(mea)
            
#             for tick in axs[i].get_xticklabels():
#                 tick.set_rotation(45)
#             axs[i].legend()
#             i = i + 1
#         plt.suptitle(type + " Data", size=16)
#         file = os.path.join(outdir, "plots.class." + type + ".png")
#         images.append(file)
#         print(f"Producing {file}")
#         plt.savefig(file, dpi=300)
#         plt.close()

#     ##############################
#     ### PRCurves
#     me = pd.merge(df_sbc, df_sr, on='srid')
#     me = me[me.metric.isin(['PRCurve_json'])]

#     types = sorted(set(me['genre']))
#     #print(sorted(set(me['class'])))
#     for clas in sorted(set(me['class'])):
#         fig, ax = plt.subplots(1, len(types), figsize=(9, 3), sharey=True)
#         fig.subplots_adjust(hspace=0.1, top=0.8)
#         i = 0
#         for type in types:
#             ax[i].set(xlim=(0, 1), xticks=np.arange(0, 1.01, 0.2),
#                       ylim=(0, 1), yticks=np.arange(0, 1.01, 0.1))
#             if (i == 0):
#                 ax[i].set_ylabel('Precision')
#             ax[i].set_xlabel('Recall')
#             ax[i].set_title(f"{type}")
#             x = me[(me.genre.isin([type]) & me['class'].isin([clas]))]
#             for index, row in x.iterrows():
#                 prc = json.loads(row['value'])
#                 ax[i].plot(prc['recall'], prc['precision'], linewidth=1.0, label=row['label'])
#             [ tick.set_size(8) for tick in ax[i].get_xticklabels() ]
#             [ tick.set_size(8) for tick in ax[i].get_yticklabels() ]
#             ax[i].legend(fontsize="6", loc ="upper right")
#             i = i + 1
#         plt.suptitle(str(clas) + " norm", size=16)        
#         file = os.path.join(outdir, "plots.prc." + str(clas) + ".png")
#         images.append(file)
#         print(f"Producing {file}")
#         plt.savefig(file, dpi=300)
#         plt.close()

#     imgs = [Image.open(x) for x in images]
#     widths, heights = zip(*(i.size for i in imgs))

#     total_width = max(widths)
#     max_height = sum(heights)

#     new_im = Image.new('RGB', (total_width, max_height))

#     y_offset = 0
#     for im in imgs:
#         new_im.paste(im, (0, y_offset))
#         y_offset += im.size[1]

#     new_im.save('test.jpg')

def copy_bestdir(df_jobs, df_sr, df_sbc, df_sa, df_sp, df_fact, param, workdir, bestdir, metric):
    me = pd.merge(df_sa, df_sr, on='srid')
    genre = 'all'
    x_ori = me[(me.metric.isin([metric]) & me.genre.isin([genre]))]
    x = x_ori.sort_values(by='value', ascending=False)
    print(x.head())
    best_label = x.iloc[0].label
    print(f"Best Selected: {best_label}")
    best_param = [ x for x in param['scoring_runs'] if x['name'] == best_label ][0]
    print(best_param)
    lev_out = os.path.join(workdir, best_param['name'])
    print(f"     Work Dir: {lev_out}")
    print(f"  copy to Dir: {bestdir}")
    os.system(f"cp {lev_out}.* {bestdir}")
    os.system(f"cp {lev_out}/* {bestdir}")
    
def main():
    parser = argparse.ArgumentParser(description='Generate a random norm/emotion submission')
    
    parser.add_argument('-C', '--code_base', type=str, required=True, default='CCU_scoring', help = 'Either the Git Repo dir or "CCU_scoring" (defualt)')
    parser.add_argument('-c', '--base_command', type=str, required=True, help = 'This is the base scoring command without the executeable name')
    parser.add_argument('-t', '--task', choices=['norm', 'emotion'], required=True, help = 'norm, emotion') ### Doesn't do anything
    parser.add_argument('-w', '--workdir', type=str, required=True, help = 'The working director`y to store results')
    parser.add_argument(      '--param_file', type=str, required=True, help = 'The param def JSON')
    parser.add_argument(      '--bestdir', type=str, default = "", help = 'The directory to copy the top run to')
    parser.add_argument('-m', '--measures', type=str, default = "average_precision f1_at_MinLLR scaled_f1_at_MinLLR", help = 'The statistics to plot - these are named for class metrics')    
     
    args = parser.parse_args()
    with open(args.param_file,"r") as f: param_txt = f.read()
    param = json.loads(param_txt)

    ### Build the factor List
    param['factors'] = { }
    for scoring_run in param['scoring_runs']:
        if ('factors' in scoring_run):
            for fact, lev in scoring_run['factors'].items():
                if (fact not in param['factors']):
                    param['factors'][fact] = []
                if (lev not in param['factors'][fact]):
                    param['factors'][fact].append(lev)    
    
    #print(json.dumps(param, indent=4), file=open(args.param_file, 'w'))
    df_jobs, df_sr, df_sbc, df_sa, df_sp, df_fact, df_sr_fact = run_score(args.code_base, args.base_command, args.workdir, param)
    measures = args.measures.split(" ")
    plot_measures_agg(df_jobs, df_sr, df_sbc, df_sa, df_sp, df_fact, df_sr_fact, measures, args.workdir)

    if (args.bestdir != ""):
        copy_bestdir(df_jobs, df_sr, df_sbc, df_sa, df_sp, df_fact, param, args.workdir, args.bestdir, metric='mean_f1_at_MinLLR')
    

if __name__ == '__main__':
	main()
