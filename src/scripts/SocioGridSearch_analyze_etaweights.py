# -*- coding: utf-8 -*-
"""

This file creates plots to analyze impact of different eta weights 
on model selection. It uses output from initial grid search run


Author: Kyla Chasalow

Last edited: August 13, 2021
"""
import sys

#import pickle

# Import relative paths
from filepaths import code_path, socio_models_path, socio_gridplots_path


# Import functions
sys.path.insert(0, code_path)#r'C:\Users\kcsky\Documents\Oxford\STATISTICS\Dissertation\GitHub\MScDissertation\src')
import LdaGridSearch
from Helpers import file_override_request


# set dpi for saving images
dpi = 250

# ask user to set whether want script to override existing figures of same name or not


fig_override = file_override_request()


        
        
# Import output of initial grid search

GridOut = LdaGridSearch.load_GridEtaTopics("GridEtaTopics_output_initialrun", path = socio_models_path)



# Set-up a Scaler object for use in Grid Eta functions
# Scaler object needs to be trained on all the output, even if a analysis of a particular Grid search over eta
# values for fixed K only uses output values for that K
output_scaler = LdaGridSearch.Scaler(LdaGridSearch._get_flat_values(GridOut, metric = "coherence"),
                           LdaGridSearch._get_flat_values(GridOut, metric = "kl"))


#options to use throughout
scalertype = "median_scaler"
aggregation_method = "median"



# Visualize the grid search over eta values for each of the K values
print("creating eta grid search plots...")

#first visualize one example individually, more close up
_ = LdaGridSearch.visualize_weighted_eta(output = GridOut[20],
                                      Scaler = output_scaler,
                                      scalertype = scalertype,
                                      aggregation_method = aggregation_method,
                                      num_weights = 10,
                                      zoom = 0,
                                      figsize = (15,10),
                                      set_figsize = True, 
                                      plot_legend = True, 
                                      plot_title = True,
                                      top_text_rotation = 45, 
                                      plot_annotation = True, 
                                      annotation_color = "black",
                                      plot_xlabel = True, 
                                      plot_ylabel = True,
                                      plot_short_title = True,
                                      save_fig = True,
                                      fig_override = fig_override,
                                      fig_name = "socio_WeightedEtaSearch_20",
                                      fig_outpath = socio_gridplots_path)



#then visualize all of them in grids
LdaGridSearch.visualize_weighted_eta_plots_grid(GridOut, 
                                               Kvals = [5, 10, 15, 20],
                                               aggregation_method = aggregation_method,
                                               scalertype = scalertype,
                                               num_weights = 15, 
                                               zoom = 0,
                                               top_text_rotation = 45,
                                               save_fig = True,
                                               fig_outpath = socio_gridplots_path,
                                               fig_override = fig_override,
                                               dpi = dpi,
                                               fig_name = "socio_WeightedEtaPlots_5-20")


LdaGridSearch.visualize_weighted_eta_plots_grid(GridOut, 
                                                Kvals = [25, 30, 35, 45],
                                               aggregation_method = aggregation_method,
                                               scalertype = scalertype,
                                               num_weights = 15, 
                                               zoom = 0,
                                               top_text_rotation = 45,
                                               plot_suptitle = False,
                                               save_fig = True,
                                               fig_outpath = socio_gridplots_path,
                                               fig_override = fig_override,
                                               dpi = dpi,
                                               fig_name = "socio_WeightedEtaPlots_25-45")







print("Creating some per-topic-metric plots...")
# Visualize per-topic metric scatterboxes for some models

#Note: these are for labelling purposes only and have to be set when running grid search
topn_phf = 25
topn_coherence = 10
thresh = 0.01


for k in [15,20]:
    _ = LdaGridSearch.GridEta_scatterbox(GridOut[k],
                                          metric = "all",
                                          topn_phf = topn_phf,
                                          topn_coherence = topn_coherence,
                                          thresh = thresh,                                         
                                          save_fig = True,
                                          fig_outpath = socio_gridplots_path,
                                          fig_name = "socio_GridEtaScatterbox_%d" %k,
                                          fig_override = fig_override,
                                          dpi = dpi
                            )
    


print("Done")
