# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 11:30:35 2021

Helper script containing file paths for navigating to rest of
project relative to /scripts directory

@author: kyla
"""
import os


#get absolute path of this file
absolute_path = os.path.abspath("filepaths")

#get to its parent (scripts)
scripts_directory = os.path.dirname(absolute_path)

#get to its grandparent (src)
src_directory = os.path.dirname(scripts_directory)

#main
main_directory = os.path.dirname(src_directory)

#NAVIGATE TO WHERE CODE STORED
code_path = os.path.join(src_directory, "func")


### Demography specific

#data
demog_data_path = os.path.join(main_directory, "data\Demography")

#models
demog_models_path = os.path.join(main_directory, "results\Demography\models")
demog_models_convergence_path = os.path.join(demog_models_path, "convergenceanalysis")
demog_models_matrix_path = os.path.join(demog_models_path, "matrices")
demog_models_randstate_path = os.path.join(demog_models_path, "randomstate_experiment")


#plots
demog_plots_path = os.path.join(main_directory, "results\Demography\plots")
demog_gridplots_path = os.path.join(demog_plots_path, "gridsearchplots")
demog_convergence_plots = os.path.join(demog_plots_path, "convergenceplots")
demog_topicword_plots = os.path.join(demog_plots_path, "topicwordplots")
demog_topicanalysis_plots = os.path.join(demog_plots_path, "topicanalysisplots")
demog_eda_plots = os.path.join(demog_plots_path, "edaplots")



### SOCIOLOGY specific 


#data
socio_data_path = os.path.join(main_directory, "data\Sociology")


#models
socio_models_path = os.path.join(main_directory, "results\Sociology\models")
socio_models_convergence_path = os.path.join(socio_models_path, "convergenceanalysis")
socio_models_matrix_path = os.path.join(socio_models_path, "matrices")

#plots
socio_plots_path = os.path.join(main_directory, "results\Sociology\plots")
socio_gridplots_path = os.path.join(socio_plots_path, "gridsearchplots")
socio_convergence_plots = os.path.join(socio_plots_path, "convergenceplots")
socio_topicword_plots = os.path.join(socio_plots_path, "topicwordplots")
socio_topicanalysis_plots = os.path.join(socio_plots_path, "topicanalysisplots")
socio_eda_plots = os.path.join(socio_plots_path, "edaplots")






