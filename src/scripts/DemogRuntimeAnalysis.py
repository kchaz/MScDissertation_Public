# -*- coding: utf-8 -*-
"""
This script does a quick analysis of computing times for training
models with different parameter settings

Author: Kyla Chasalow
Last edited: August 2, 2021
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Import relative paths
from filepaths import code_path
from filepaths import demog_convergence_plots as plot_path
from filepaths import  demog_models_convergence_path as model_path1
from filepaths import demog_models_path as model_path2


# Import functions
sys.path.insert(0, code_path)
import LdaLogging
from Helpers import figure_saver








# GRID PLOT FOR ALL THE LDA MODELS FIT WITH LOGGING


plt.rcParams.update({'font.family':'serif'})
fig, ax = plt.subplots(nrows = 2, ncols=2, sharex=False, sharey=True, figsize=(15,15))
fig.text(-.07, 0.5, "Time (seconds)", va='center', rotation='vertical', fontsize = 30)
fig.tight_layout(h_pad=13, w_pad = 10)


#times for varying epoch
epoch_vals = [0,1,2,10,20,30,40,50,60]
fnames = [os.path.join(model_path1, "convergence_test_%d_200.log" % e) for e in epoch_vals]
epoch_times = LdaLogging.get_all_log_durations(fnames)

#times for varying K
Kvals = [5, 10, 25, 50]
fnames = [os.path.join(model_path1, "init_topic_test_%d.log") %k for k in Kvals]
K_times = LdaLogging.get_all_log_durations(fnames)


#times for varying eta
etas = [0.01,0.1,1,10,100]
fnames = [os.path.join(model_path1, "init_eta_test_%s.log" % str(eta)) for eta in etas]
eta_times = LdaLogging.get_all_log_durations(fnames)

#times for varying number of iterations
num_iter = [100,150,200,250]
fnames = [os.path.join(model_path1, "convergence_test_60_%d.log" %n) for n in num_iter]
iter_times = LdaLogging.get_all_log_durations(fnames)

max_val = np.max([np.max(elem) for elem in [epoch_times, K_times, eta_times, iter_times]]) 

plt.subplot(2,2,1)
LdaLogging.comptime_comparison_plot(num_iter, iter_times, 
                        xlabel = "Number of iterations", 
                        title = "Time vs Per-Doc Iterations \n (epochs = 60, K = 10, $\eta$ = 0.1)",
                        set_figsize = False,
                        plot_ylabel = False,
                        ylim = (0,max_val * 1.1))

plt.subplot(2,2,2)
LdaLogging.comptime_comparison_plot(epoch_vals, 
                                   epoch_times, 
                                   xlabel  = "Epochs",
                                   title = "Times vs Training Epochs \n (iterations = 200, K = 10, $\eta$ = 0.1)",
                                   set_figsize = False,
                                   plot_ylabel = False,
                                   ylim = (0,max_val * 1.1))


plt.subplot(2,2,3)
LdaLogging.comptime_comparison_plot(etas, eta_times, 
                        xlabel = "$\eta$", 
                        title = "Time vs $\eta$ \n (epochs = 40, iterations = 200, K = 10)",
                        set_figsize= False,
                        plot_ylabel = False,
                        ylim = (0,max_val * 1.1))


plt.subplot(2,2,4)
LdaLogging.comptime_comparison_plot(Kvals, 
                                   K_times, 
                                   xlabel = "K",
                                   title = "Time vs K \n (epochs = 40, iterations = 200, $\eta$ = 0.1)",
                                   set_figsize = False,
                                   plot_ylabel = False,
                                   ylim = (0,max_val * 1.1)
                                  )
plt.suptitle("Computing Times for Models with Logging ", fontsize = 35)
plt.subplots_adjust(top = .86)     


figure_saver(fig_name = "Demog_comptime_analysis1", 
                 outpath = plot_path,
                 dpi = 200,
                 fig_override = True,
                 bbox_inches = "tight")


print("finished grid...")



#### PLOT FOR BEST K MODELS

Kvals2 = [5,10,15,20,25,30,35,40,50,60]
fnames = [os.path.join(model_path2, "Best_%d.log" %k) for k in Kvals2]
K_times2 = LdaLogging.get_all_log_durations(fnames)
LdaLogging.comptime_comparison_plot(Kvals2, K_times2, 
                        xlabel = "K", 
                        title = "Computing times vs K with no logging \n (epochs = 30, iterations = 200, variable $\eta$)",
                        xtick_rotation = 0,
                        set_xticks_to_xvals = True,
                        figsize = (12,6))

figure_saver(fig_name = "Demog_comptime_analysis2", 
                 outpath = plot_path,
                 dpi = 200,
                 fig_override = True,
                 bbox_inches = "tight")


print("done")

