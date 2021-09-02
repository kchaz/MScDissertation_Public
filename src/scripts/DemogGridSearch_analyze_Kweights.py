# -*- coding: utf-8 -*-
"""

This file creates plots to analyze impact of different K weights 
on model selection. It uses output from initial grid search run

Author: Kyla Chasalow
Last edited: July 21, 2021
"""
import sys


# Import relative paths
from filepaths import code_path, demog_models_path, demog_gridplots_path


# Import functions
sys.path.insert(0, code_path)#r'C:\Users\kcsky\Documents\Oxford\STATISTICS\Dissertation\GitHub\MScDissertation\src')
import LdaGridSearch
from Helpers import file_override_request


# set dpi for saving images
dpi = 250

# ask user to set whether want script to override existing figures of same name or not

fig_override = file_override_request()



# set eta weights after examining plots output by DemogGridSearch_analyze_etaweights.py
eta_weights = (.75,.25)

        
        
# Import output of initial grid search

GridOut = LdaGridSearch.load_GridEtaTopics("GridEtaTopics_output_initialrun", path = demog_models_path)



#Visualize grid search over number of topics
print("creating topics grid search plots...")
out = LdaGridSearch.visualize_weighted_eta_topics(GridOut, 
                                                use_same_weights = False,
                                                eta_weights = eta_weights,
                                                scalertype = "median_scaler",
                                                aggregation_method = "median", 
                                                num_weights = 40,
                                                zoom = 0,
                                                figsize = (15,10), 
                                                set_figsize = True, 
                                                plot_legend = True, 
                                                plot_title = True,
                                                save_fig = True,
                                                fig_outpath = demog_gridplots_path,
                                                fig_name = "WeightedEtaTopicPlot",
                                                fig_override = fig_override,
                                                dpi = dpi)


#visualize per-topic-metrics grid
#Note: these are for labelling purposes only and have to be set when running grid search
topn_phf = 25
topn_coherence = 10
thresh = 0.01


output = LdaGridSearch.WeightedEtaTopicsSearch(GridOut, 
                                scalertype = "median_scaler",
                                aggregation_method = "median",
                                eta_weights = eta_weights)

_ = LdaGridSearch.GridEtaTopics_scatterbox(output, 
                                           metric = "all", 
                                           topn_coherence = topn_coherence,
                                           topn_phf = topn_phf,
                                           thresh = thresh,
                                           xtick_rotation = 45,
                                           save_fig = True,
                                           fig_outpath = demog_gridplots_path,
                                           fig_name = "GridEtaTopicsScatterbox",
                                           fig_override = fig_override,
                                           dpi = dpi)


# print information about the exact order of the models at a few weights

weights_to_examine = [(.87,.13),(.88, .12),(.89,.11), (.90,.10),
                      (.91,.09), (.92,.08), (.93,.07), (.94,.06)]

for w in weights_to_examine:
    best_dict = LdaGridSearch.get_overall_best(output, 
                                               K_weights = w,
                                               scalertype = "median_scaler",
                                               aggregation_method = "median",
                                               save_best = False)
    
    print("Weight (%s,%s) model order (best to worst): " % (str(w[0]), str(w[1])), best_dict["all_model_rankings"])

