# -*- coding: utf-8 -*-
"""
This file creates plots for an initial analysis of convergence 
properties

Requires DemogConvergenceAnalysis_models.py to have been run

Author: Kyla Chasalow
Last edited: August 31, 2021
"""

import os
import sys
import pickle
import seaborn as sns

# Import relative paths
from filepaths import code_path
from filepaths import  demog_models_convergence_path as model_path
from filepaths import demog_convergence_plots as plot_path


# Import functions
sys.path.insert(0, code_path)
import LdaLogging 
from Helpers import file_override_request
import LdaOutputPerTopicMetrics
import LdaOutputTopicSimilarity

# ask user to set whether want script to override existing figures of same name or not
fig_override = file_override_request()

#set dpi for saving figures
dpi = 150



#NOTE: THESE ONLY AFFECT THE PLOT LABELLING - THEIR ACTUAL VALUES 
#MUST BE SET WHEN TRAINING THE MODELS.
topn_coherence = 10
topn_phf = 25
thresh = 0.01



print("Creating plots from Experiment 1")

num_topics = 10
alpha = "auto"
eta = 0.1
passes = 60
num_iter = [100, 150, 200, 250]
#file names
fnames = ["convergence_test_" + str(passes) + "_" + str(it) for it in num_iter]


#create metric comparison plot using the logs of the models just created
_ = LdaLogging.metric_comparison_plot(fnames, "all",
                                         path = model_path,
                                         labels = ["100 iterations", "150 iterations", "200 iterations", "250 iterations"],
                                         save_fig = True,
                                         fig_outpath = plot_path,
                                         fig_name = "ConvergencePlot_%d_%s" % (num_topics, str(eta)),
                                         fig_override = fig_override,
                                         dpi = dpi)







print("Creating plots from Experiment 2")
#scatterboxes of topic metrics over epochs


with open(os.path.join(model_path,"epoch_test_summary_array.pickle"), "rb") as handle:   
     summary_array = pickle.load(handle) 
            
label_list = ["0","1","2","10","20","30","40","50","60"]
LdaOutputPerTopicMetrics.visualize_spread(summary_array, 
                            metric = "all", 
                            topn_coherence = topn_coherence, 
                            topn_phf = topn_phf, 
                            thresh = thresh,
                            labels = label_list, 
                            xlabel = "Epoch", 
                            plot_points = True,  
                            sup_title = "Comparing Topic Metrics Over Epochs",
                            save_fig = True,
                            fig_outpath = plot_path,
                            fig_name = "scatterbox_varyingepochs",
                            fig_override = fig_override,
                            dpi = dpi)



with open(os.path.join(model_path,"topic_dif_over_epoch_array.pickle"), "rb") as handle:   
     diff_list = pickle.load(handle) 

_ = LdaOutputTopicSimilarity.visualize_topic_difs(diff_list,
                                  distance = "jensen_shannon",
                                  labels =  ["0","1","2","10","20","30","40","50","60"],
                                  xlabel = "Training Epochs",
                                  alpha_point = 0.2,
                                  save_fig = True,
                                  fig_outpath = plot_path,
                                  fig_name = "Scatterbox_topic_dif_varyingepochs",
                                  fig_override = fig_override,
                                  dpi = dpi)







print("Creating plots from Experiment 3")


with open(os.path.join(model_path,"K_test_summary_array.pickle"), "rb") as handle:   
     summary_array = pickle.load(handle) 


#figure out whether can use the log for the 10-topic model from exp1
#or else whether can use from run of exp 3 (or else have a problem)
try:
    LdaLogging.load_log("convergence_test_40_200.log", model_path)
    num_topics = [5,25,50]
    fnames = ["init_topic_test_" + str(k) for k in num_topics]  
    fnames = [fnames[0]] + ["convergence_test_40_200"] +  fnames[1:3]        
except:
    LdaLogging.load_log("init_topic_test_10.log", model_path)
    num_topics = [5,10,25,50]
    fnames = ["init_topic_test_" + str(k) for k in num_topics] 

# Convergence plots from logs
_ = LdaLogging.metric_comparison_plot(fnames, 
                                         metric = "all",
                                         path = model_path,
                                         labels = ["5 Topics","10 Topics","25 Topics","50 Topics"],
                                         save_fig = True,
                                         fig_outpath = plot_path,
                                         fig_name = "ConvergencePlot_varyingK",
                                         fig_override = fig_override,
                                         dpi = dpi)



#look at mean coherence vs K value for last epoch of each model
logs = [LdaLogging.load_log(os.path.join(model_path,f + ".log")) for f in fnames]
LdaLogging.plot_mean_coherence_comparison(logs, epoch = None, connect = False,
                                          save_fig = True,
                                          fig_outpath = plot_path,
                                          fig_name = "MeanCoherenceComparison",
                                          fig_override = fig_override,
                                          dpi = dpi)




labels = ["5","10","25","50"]
LdaOutputPerTopicMetrics.visualize_spread(summary_array, 
                               metric = "all", 
                               labels = labels,
                               topn_coherence = topn_coherence, 
                               topn_phf = topn_phf, 
                               thresh = thresh,
                               xlabel = "Number of Topics (K)", plot_points = True, 
                               alpha_point = .45, alpha_box = .3,
                               color_point = "blue", 
                               sup_title = "Comparing Topic Metrics (Varying K,  $\eta = .01$)",
                               save_fig = True,
                               fig_outpath = plot_path,
                               fig_name = "scatterbox_varyingK",
                               fig_override = fig_override,
                               dpi = dpi)


# look at relationships between topic metrics
colors = [sns.color_palette("bright")[i] for i in [0,2,4,1]]
test = LdaOutputPerTopicMetrics.multi_model_pairplot(summary_array,
                                     labels = ["5 Topic","10 Topic","25 Topic","50 Topic"], 
                                     colors = colors, 
                                     alpha = .5,
                                     save_fig = True,
                                     fig_outpath = plot_path,
                                     fig_name = "topicmetricrelations",
                                     fig_override = fig_override,
                                     dpi = dpi)
    


with open(os.path.join(model_path,"topic_dif_by_K_array.pickle"), "rb") as handle:   
     diff_list = pickle.load(handle) 

labels = ["5","10","25","50"]
_ = LdaOutputTopicSimilarity.visualize_topic_difs(diff_list,
                                   distance = "jensen_shannon",
                                   labels =  labels,
                                   xlabel = "Number of Topics (K)",
                                   alpha_point = 0.2,
                                   save_fig = True,
                                   fig_outpath = plot_path,
                                   fig_name = "Scatterbox_topic_dif_varyingK",
                                   fig_override = fig_override,
                                   dpi = dpi)





print("Creating plots from Experiment 4")

with open(os.path.join(model_path,"eta_test_summary_array.pickle"), "rb") as handle:   
     summary_array = pickle.load(handle) 


#key parameters
alpha = "auto"
num_topics = 10
passes = 40
iterations = 200

etas = [.01, .1, 1, 10, 100]     
fnames = ["init_eta_test_" + str(eta) for eta in etas]

labels = [r"$\eta$ = " + str(eta) for eta in etas ]
_ = LdaLogging.metric_comparison_plot(fnames,
                                         path = model_path,
                                         labels =  labels,
                                         metric = "all",
                                         save_fig = True,
                                         fig_outpath = plot_path,
                                         fig_name = "ConvergencePlot_varyingeta",
                                         fig_override = fig_override,
                                         dpi = dpi)



eta_labels = [".01",".1","1","10","100"]
LdaOutputPerTopicMetrics.visualize_spread(summary_array,
                           metric = "all",
                                  topn_coherence = topn_coherence, 
                                  topn_phf = topn_phf, 
                                  thresh = thresh,
                                  labels = eta_labels, 
                                  xlabel = "$\eta$", 
                                  plot_points = True,  
                                  sup_title = "Comparing Topic Metrics (K = 10, Varying $\eta$)",
                                  save_fig = True,
                                  fig_outpath = plot_path,
                                  fig_name = "scatterbox_varyingeta",
                                  fig_override = fig_override,
                                  dpi = dpi)
