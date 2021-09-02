# -*- coding: utf-8 -*-
"""

This file creates bar plots of topics for the best model for each K
and saves them to a model specific folder


Author: Kyla Chasalow
Last edited: September 2, 2021
"""
RUNALL = True #if True, generates all topic plots for all models

#standard imports
import sys
import pickle
import os
import logging
import numpy as np
import matplotlib.pyplot as plt


#gensim imports
from gensim.models.ldamodel import LdaModel

# Import relative paths
from filepaths import code_path
from filepaths import socio_data_path as data_path
from filepaths import  socio_models_path as model_path
from filepaths import socio_models_matrix_path as matrix_path
from filepaths import socio_topicword_plots as plot_path


# Import functions
sys.path.insert(0, code_path)#r'C:\Users\kcsky\Documents\Oxford\STATISTICS\Dissertation\GitHub\MScDissertation\src')
import AbstractCleaner
import LdaOutput
import LdaOutputWordPlots
from Helpers import file_override_request


# ask user to set whether want script to override existing figures of same name or not
message_1 = "Override existing figure files in destination folder(s) that have same name? (Yes/No):"
fig_override = file_override_request(message_1)
message_2 = "\n \n \n Re-generate existing theta matrices, even if already exist?"
message_2 += "\n Note: if they do not already exist, I will generate them either way"
message_2 += "\n Warning: because of randomness involved, matrices will likely be slightly different from previous run (Yes/No):"
theta_override = file_override_request(message_2)


#set dpi for saving figures
dpi = 150


# Load processed abstracts and get corpus and dictionary
filepath = os.path.join(data_path, "Socio_abstracts4_FreqFilter.pickle")
with open(filepath, "rb") as fp:   #Unpickling
    cleaned_docs = pickle.load(fp)
print(len(AbstractCleaner.extract_vocab(cleaned_docs)), "words in vocab")
print(AbstractCleaner.corpus_length(cleaned_docs), "words in corpus")
corpus, dictionary = AbstractCleaner.get_corpus_and_dictionary(cleaned_docs)



# SET UP LOGGING TO CONSOLE
#create instance completely separate from other log
logger = logging.getLogger() 
#create handler and set its properties
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s : %(levelname)s - %(message)s")
handler.setFormatter(formatter)
#add handler to logger
logger.addHandler(handler) 



#LOAD ALL THE MODELS
Kvals = [5, 10, 15, 20, 25, 30 ,35, 40, 45]
fnames = ["Best_%d_model"%k for k in Kvals]
model_dict = {}
for i,k in enumerate(Kvals):
    model_dict[k] = LdaModel.load(os.path.join(model_path,fnames[i]))
    
    
#file names for theta matrices
fnames = ["theta_matrix_" + str(k) for k in Kvals]


def fit_matrices():
    """Quick helper to avoid typing this twice below"""
    for k, f in zip(Kvals, fnames):
        wordtopics = False
        #generate per-document array of expected counts per word-topic combo for only a few
        if k in [15,20,25,30]:
            wordtopics = True
        LdaOutput.get_document_matrices(model = model_dict[k],
                                               corpus = corpus,  
                                               minimum_probability = .01, #NOTE: somewhat arbitrary choice
                                               save_theta_matrix = True, 
                                               theta_outpath = matrix_path,
                                               theta_filename = f,
                                               per_word_topics = wordtopics,
                                               save_wordtopic_arrays = wordtopics,
                                               dictionary = dictionary,
                                               minimum_phi_value = .01,
                                               wordtopic_filename = "wordtopic_arrays_%d" %k,
                                               wordtopic_outpath = matrix_path,
                                               override = True                                      
                                          )



      
# Load saved theta matrices into dictionary
#if override, then in any case just fit matrices
if theta_override:
     fit_matrices()
 
    
#try loading existing matrices
try: 
    theta_dict = {}
    for i, k in enumerate(Kvals):
        theta_dict[k] = np.load(os.path.join(matrix_path, fnames[i] + ".npy"))
#if don't exist, create matrices and then load them into dictionary
except:
    print("matrices do not yet exist. Creating matrices")
    fit_matrices(override = theta_override)    #only override
    theta_dict = {}
    for i, k in enumerate(Kvals):
        theta_dict[k] = np.load(os.path.join(matrix_path, fnames[i] + ".npy"))


#Get all word plots
if RUNALL:
    for k in Kvals:
        
        print("%d-topic plot" % k)
        
        path = os.path.join(plot_path, "%dtopic"%k)
        
        LdaOutputWordPlots.topic_relevance_grid(model = model_dict[k],
                              corpus = corpus,
                              dictionary = dictionary,
                              value_type = "counts",
                              theta_mat = theta_dict[k], 
                              lamb = 0.6,
                              topn = 20,
                              minimum_probability = .01, 
                              first_title_only = False,
                              save_all_plots = True, 
                              custom_name = "socio_%d_topic_grid" %k,
                              dpi = dpi, 
                              fig_outpath = path,
                              fig_override = fig_override,
                              display_figure = False
                              )
    
        plt.close('all') #close to avoid consuming too much memory by leaving all these plots open
   
    





#### A FEW SPECIFIC PLOTS 

#networks
LdaOutputWordPlots.topic_relevance_grid(model = model_dict[20],
                          corpus = corpus,
                          dictionary = dictionary,
                          plot_all_topics = False,
                          custom_list = [12, 17],
                          value_type = "counts",
                          theta_mat = theta_dict[20], 
                          lamb = 0.6,
                          topn = 20,
                          plot_title_and_legend = True,
                          custom_title = "20-Topic Model \n", #" \n (ordered by relevance, $\lambda = 0.6$)",
                          save_all_plots = True, 
                          dpi = dpi, 
                          custom_name = "socio_20_networks",
                          fig_outpath = os.path.join(plot_path, "20topic"),
                          first_title_only = False,
                          fig_override = fig_override,
                          display_figure = False,
                          )

#Race/ethnicity
LdaOutputWordPlots.topic_relevance_grid(model = model_dict[20],
                          corpus = corpus,
                          dictionary = dictionary,
                          plot_all_topics = False,
                          custom_list = [1,6],
                          value_type = "counts",
                          theta_mat = theta_dict[20], 
                          lamb = 0.6,
                          topn = 20,
                          plot_title_and_legend = True,
                          custom_title = "20-Topic Model \n", #" \n (ordered by relevance, $\lambda = 0.6$)",
                          save_all_plots = True, 
                          dpi = dpi, 
                          custom_name = "socio_20_raceethnicity",
                          fig_outpath = os.path.join(plot_path, "20topic"),
                          first_title_only = False,
                          fig_override = fig_override,
                          display_figure = False,
                          )


#review/theory vs empirical/surveys
LdaOutputWordPlots.topic_relevance_grid(model = model_dict[20],
                          corpus = corpus,
                          dictionary = dictionary,
                          plot_all_topics = False,
                          custom_list = [3,5],
                          value_type = "counts",
                          theta_mat = theta_dict[20], 
                          lamb = 0.6,
                          topn = 20,
                          plot_title_and_legend = True,
                          custom_title = "20-Topic Model \n", #" \n (ordered by relevance, $\lambda = 0.6$)",
                          save_all_plots = True, 
                          dpi = dpi, 
                          custom_name = "socio_20_methodological",
                          fig_outpath = os.path.join(plot_path, "20topic"),
                          first_title_only = False,
                          fig_override = fig_override,
                          display_figure = False,
                          )


#additional network topic
i=7
LdaOutputWordPlots.topic_relevance_barplot(model = model_dict[20],
                            topicid = i, 
                            corpus = corpus,
                            dictionary = dictionary,
                            value_type = "counts",
                            theta_mat = theta_dict[20],
                            lamb = 0.6,
                            topn = 20,
                            figsize = (6,7),
                            save_fig = True,
                            fig_name = "socio_20topic_%d" % i,
                            fig_outpath = os.path.join(plot_path, "20topic"),
                            dpi = dpi,
                            fig_override = fig_override
                            )


