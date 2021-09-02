# -*- coding: utf-8 -*-
"""

Previous script: SocioTopicWordPlots.py
Assumes this has been run so that theta matrices already exist


This file analyzes the topics of the final model(s)


Author: Kyla Chasalow
Last edited: August 31, 2021
"""


#standard imports
import sys
import pickle
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#gensim imports
from gensim.models.ldamodel import LdaModel

# Import relative paths
from filepaths import code_path
from filepaths import socio_data_path as data_path
from filepaths import socio_models_path as model_path
from filepaths import socio_models_matrix_path as matrix_path
from filepaths import socio_topicword_plots as wordplot_path

# Import functions
sys.path.insert(0, code_path)#r'C:\Users\kcsky\Documents\Oxford\STATISTICS\Dissertation\GitHub\MScDissertation\src')
import AbstractCleaner
import LdaOutput
import LdaOutputPerTopicMetrics
import LdaOutputTopicSimilarity
import LdaOutputTopicSizePlots
import LdaOutputTimePlots
import LdaOutputGroupPlots
import LdaOutputDocs
from Helpers import file_override_request



# ask user to set whether want script to override existing figures of same name or not
fig_override = file_override_request()


#set dpi for saving figures
dpi = 150


#Load data and extract key pieces
TM_data_final = pd.read_csv(os.path.join(data_path,"Socio_data_final.csv"))
journals = list(TM_data_final["prism:publicationName"])
year_labels = TM_data_final.Year


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
    

      
# Load saved theta matrices into dictionary
fnames = ["theta_matrix_" + str(k) + ".npy" for k in Kvals]
theta_dict = {}
for i, k in enumerate(Kvals):
    theta_dict[k] = np.load(os.path.join(matrix_path, fnames[i]))

#load word-topic arrays - I only use 20 below
vals = [20]
fnames = ["wordtopic_arrays_%d.npz"%k for k in vals]
wordtopic_dict = {}
for i, k in enumerate(vals):
    wordtopic_dict[k] = LdaOutput.load_wordtopic_array(filename = fnames[i], path = matrix_path)
    




###############################################################################
###############################################################################

#Coherence of individual topics for each model

for k in Kvals:
    if k <= 10:
        figsize = (10, 6)
    if k <= 20:
        figsize = (10, 8)
    elif k <= 30:
        figsize = (10,10)
    else:
        figsize = (10,18)
    LdaOutputPerTopicMetrics.topic_metric_order_plot(model_dict[k], 
                                                      corpus = corpus, 
                                                      dictionary = dictionary,
                                                      metric = "coherence", 
                                                      topn_coherence = 20,
                                                      figsize = figsize,
                                                      save_fig = True,
                                                      fig_name = "socio_%d_topic_order_plot" %k,
                                                      fig_outpath = os.path.join(wordplot_path, "%dtopic"%k),
                                                      fig_override = fig_override,
                                                      dpi = dpi)
    
    plt.close('all') #close to avoid consuming too much memory by leaving all these plots open

    
# Topic sizes for each model overall and by journal
doc_lengths = LdaOutput.get_doc_lengths(corpus)

for k in Kvals:
    word_counts = LdaOutput.get_topic_wordcounts(theta_dict[k], doc_lengths, normalized = False)
    doc_counts = LdaOutput.get_topic_doccounts(theta_dict[k])
    means = LdaOutput.get_topic_means(theta_dict[k])

    df = LdaOutputTopicSizePlots.topic_size_comparison_plot(word_counts = word_counts,
                        doc_counts = doc_counts,
                        medians = None,
                        means = means,
                        plot_type = "barplot",
                        figsize = (16, 6),
                        save_fig = True,
                        fig_name = "socio_%dTopic_Sizes" %k,
                        fig_outpath =  os.path.join(wordplot_path, "%dtopic"%k),
                        fig_override = fig_override,
                        dpi = dpi)
    
    
    
    per_group_size_dict = LdaOutput.get_per_group_topic_size(theta_mat = theta_dict[k], 
                                       label_list = journals,
                                       sizetype = "word_count", 
                                       doc_lengths = doc_lengths,
                                       normalized = True) #doesn't make sense to plot this without normalizing

    LdaOutputGroupPlots.plot_topic_size_by_group(per_group_size_dict, 
                                                 "barplot_by_topic",
                                                 group_name = "Journal",
                                                legend_outside_plot = False,
                                                save_fig = True,
                                                fig_name = "socio_byjournal_%dTopic_Sizes1" %k,
                                                fig_outpath =  os.path.join(wordplot_path, "%dtopic"%k),
                                                fig_override = fig_override,
                                                dpi = dpi)
    
    LdaOutputGroupPlots.plot_topic_size_by_group(per_group_size_dict, 
                                                "barplot_by_group",
                                                group_name = "Journal",
                                                plot_topic_colors = False,
                                                save_fig = True,
                                                fig_name = "socio_byjournal_%dTopic_Sizes2" %k,
                                                fig_outpath =  os.path.join(wordplot_path, "%dtopic"%k),
                                                fig_override = fig_override,
                                                dpi = dpi)

    
    plt.close('all') #close to avoid consuming too much memory by leaving all these plots open
       


# DOCUMENT SPLIT PLOT FOR TOPIC 5.4
LdaOutputDocs.plot_doc_trajectories(theta_dict = theta_dict,
                      root_k = 5,
                      topic_id = 4,
                      plot_all = False,
                      min_K = None,
                      max_K = 25,
                      single_color = "purple",
                      cmap = "tab20", 
                      figsize = (15,20),
                      shift_width = True,
                      save_fig = True,
                      fig_outpath = os.path.join(wordplot_path, "5topic"),
                      fig_name = "Socio5.4_doc_trajectories",
                      dpi = dpi,
                      fig_override = fig_override)
               

#-------------------------------------------------------------------------------
# 20-TOPIC MODEL
#-------------------------------------------------------------------------------

# Correlation matrix heatmap
_ = LdaOutputTopicSimilarity.topic_correlations(theta_dict[20],
                                                plot = True, 
                                                cmap = "PuOr",
                                                save_fig = True, 
                                                fig_outpath = os.path.join(wordplot_path, "20topic"),
                                                fig_name = "socio_20_topic_correlation_map", 
                                                dpi = dpi,
                                                fig_override = fig_override)


# Similarity Matrix heatmap
_ = LdaOutputTopicSimilarity.plot_topic_heatmap(qmodel = model_dict[20],
                                                pmodel = model_dict[20], 
                                                distance = "jensen_shannon", 
                                                normed = False, 
                                                save_fig = True,
                                                fig_outpath = os.path.join(wordplot_path, "20topic"),
                                                fig_name = "socio_20_topic_similarity_map",
                                                dpi = dpi,
                                                fig_override = fig_override)



# plot looking at how topic size tracks with other topic metrics
word_counts = LdaOutput.get_topic_wordcounts(theta_dict[20], doc_lengths, normalized = False)
LdaOutputTopicSizePlots.metric_size_comparison_grid(model = model_dict[20], 
                            corpus = corpus,
                            dictionary = dictionary,
                            sizes = word_counts,
                            label = "Expected Word Counts",
                            save_fig = True,
                            fig_outpath = os.path.join(wordplot_path, "20topic"),
                            fig_name = "socio_20topic_metricsizeplot",
                            dpi = dpi,
                            fig_override = fig_override)









print("Document proportion histograms")

for i in [3,5,7,12]:
    LdaOutputDocs.theta_hist_by_group(theta_dict[20],
                        group_list = journals,
                        topic_id = i,
                        color = "green",
                        alpha = 0.5,
                        bins = 25,
                        group_name = "Journal",
                        remove_zeros = False,
                        normalize = True,
                        save_fig = True,
                        fig_outpath = os.path.join(wordplot_path, "20topic"),
                        fig_name = "socio_20topic_%d_theta_dist"%i,
                        dpi = dpi,
                        fig_override = fig_override)

#zoom in on 12
LdaOutputDocs.theta_hist_by_group(theta_dict[20],
                       group_list = journals,
                       topic_id = 12,
                       color = "green",
                       alpha = 0.5,
                       bins = 25,
                       title = "\nZero-truncated",
                       remove_zeros = True,
                       normalize = True,
                       group_name = "Journal",
                       save_fig = True,
                       fig_outpath = os.path.join(wordplot_path, "20topic"),
                       fig_name = "socio_20topic_12_theta_dist_zoom",
                       dpi = dpi,
                       fig_override = fig_override)





plt.close('all') #close to avoid consuming too much memory by leaving all these plots open
print("Topics over time")
#word plot just for topic 11, which shows that sporadic pattern and is highly incoherent
k=20
per_year_topics_dict = LdaOutput.get_per_group_topics(wordtopic_dict[k], 
                                                      label_list = year_labels,
                                                      normalized = True,
                                                      save_dict = False, 
                                                     )
_ = LdaOutputTimePlots.plot_topic_by_year(per_year_topics_dict, 
                   topic_id = 11,
                   model = model_dict[20],
                   corpus = corpus,
                   figsize = (20,22),
                   dictionary = dictionary,
                   lamb = 0.6, topn = 30,
                   color = "green", alpha = 0.25,
                   save_fig = True,
                   fig_outpath = os.path.join(wordplot_path, "20topic"),
                   fig_name = "socio_20topic_11wordtimeplot",
                   dpi = dpi,
                   fig_override = fig_override)


#use this to take out 2021 for the two journals that have 2021 observations
zoom_dict = {}
for g in np.unique(journals):
    zoom_dict[g] = (0,1)
   
zoom_dict["Annual Review of Sociology"] = (0,0)


# overall over time grid
LdaOutputGroupPlots.topic_by_group_time_plot(theta_dict[20],
                   group_list = journals,
                   year_list = year_labels,
                   zoom_dict = zoom_dict,
                   doc_lengths = doc_lengths,
                   sizetype = "word_count",
                   to_plot_list = "all",
                   title_group_label = "Journal",
                   legend_label = "",
                   legend_loc = (0,-.1),
                   save_fig = True,
                   fig_outpath = os.path.join(wordplot_path, "20topic"), 
                   fig_name = "socio_20_journaltimeplot_20_",
                   dpi = dpi,
                   fig_override = fig_override)
        




print("Per-group topics")
plt.close('all') #close to avoid consuming too much memory by leaving all these plots open

LdaOutputGroupPlots.plot_topic_words_by_group(wordtopic_array = wordtopic_dict[20],
                              group_list =  journals,
                              value_type = "counts",
                              to_plot_list = "all",
                              corpus = corpus,
                              dictionary = dictionary,
                              topn = 20,
                              lamb = 0.6, 
                              group_name = "Journal",
                              plot_overall_magnitudes = True,
                              save_fig = True,
                              fig_name = "socio_20_word_by_group",
                              fig_override = fig_override,
                              dpi = dpi,
                              fig_outpath = os.path.join(wordplot_path, "20topic\\topicsbyjournal"))


plt.close('all') #close to avoid consuming too much memory by leaving all these plots open


print("Tada!")