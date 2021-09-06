# -*- coding: utf-8 -*-
"""

Previous script: DemogTopicWordPlots.py
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
from filepaths import demog_data_path as data_path
from filepaths import  demog_models_path as model_path
from filepaths import demog_models_matrix_path as matrix_path
from filepaths import demog_topicanalysis_plots as plot_path
from filepaths import demog_topicword_plots as wordplot_path

# Import functions
sys.path.insert(0, code_path)#r'C:\Users\kcsky\Documents\Oxford\STATISTICS\Dissertation\GitHub\MScDissertation\src')
import AbstractCleaner
import LdaOutput
import LdaOutputPerTopicMetrics
import LdaOutputTopicSimilarity
import LdaOutputTopicSizePlots
import LdaOutputTimePlots
import LdaOutputDocs
import LdaOutputGroupPlots
from Helpers import file_override_request



# ask user to set whether want script to override existing figures of same name or not
fig_override = file_override_request()


#set dpi for saving figures
dpi = 225


#Load data and extract year labels
TM_data_final = pd.read_csv(os.path.join(data_path,"Demog_data_final.csv"))
year_labels = TM_data_final.Year 

# Load abstracts and apply last pre-processing step (reomving words in under 5 documents)
# Load processed abstracts and get corpus and dictionary
filepath = os.path.join(data_path, "Demog_abstracts4_FreqFilter.pickle")
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



Kvals = [5, 10, 15, 20, 25, 30 ,35, 40, 50, 60]

# LOAD ALL THE MODELS
fnames = ["Best_%d_model"%k for k in Kvals]
model_dict = {}
for i,k in enumerate(Kvals):
    model_dict[k] = LdaModel.load(os.path.join(model_path,fnames[i]))
    
      
# Load saved theta matrices into dictionary
fnames = ["theta_matrix_" + str(k) + ".npy" for k in Kvals]
theta_dict = {}
for i, k in enumerate(Kvals):
    theta_dict[k] = np.load(os.path.join(matrix_path, fnames[i]))

#load wordtopic arrays
doc_wordtopic_arrays20 = LdaOutput.load_wordtopic_array("wordtopic_arrays_20.npz", path = matrix_path)
doc_wordtopic_arrays25 = LdaOutput.load_wordtopic_array("wordtopic_arrays_25.npz", path = matrix_path)
doc_wordtopic_arrays30 = LdaOutput.load_wordtopic_array("wordtopic_arrays_30.npz", path = matrix_path)



###### Coherence of individual topics for each model

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
                                                      fig_name = "%d_topic_order_plot" %k,
                                                      fig_outpath = os.path.join(wordplot_path, "%dtopic"%k),
                                                      fig_override = fig_override,
                                                      dpi = dpi)


# Topic size by topic ID bar plot for each model
print("Creating topic size by topic ID bar plots for each model")
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
                        fig_name = "demog_%dTopic_Sizes" %k,
                        fig_outpath =  os.path.join(wordplot_path, "%dtopic"%k),
                        fig_override = fig_override,
                        dpi = dpi)

    
    plt.close('all') #close to avoid consuming too much memory by leaving all these plots open
       




#Compare measures of topic size using scatterplot for 20-topic model
print("Creating topic size scatterplot for 20-topic model")

k = 20  #change this to examine other models - make sure to change the file names to be saved, too!
doc_lengths = LdaOutput.get_doc_lengths(corpus)
word_counts = LdaOutput.get_topic_wordcounts(theta_dict[k], doc_lengths, normalized = False)
doc_counts = LdaOutput.get_topic_doccounts(theta_dict[k])
means = LdaOutput.get_topic_means(theta_dict[k])
medians = LdaOutput.get_topic_medians(theta_dict[k])
print("Medians are", medians) #not useful, all 0...

#plot comparing the different topic metrics
df = LdaOutputTopicSizePlots.topic_size_comparison_plot(word_counts = word_counts,
                    doc_counts = doc_counts,
                    medians = None,
                    means = means,
                    plot_type = "scatterplot",
                    plot_title = True, 
                    figsize = (16, 6),
                    nbins = 7,
                    save_fig = True,
                    fig_name = "20Topic_SizeComparison",
                    fig_outpath = plot_path,
                    fig_override = fig_override,
                    dpi = dpi)


plt.close('all') #close to avoid consuming too much memory by leaving all these plots open
  




#plot looking at how topic size tracks with other topic metrics
print("Creating plot examining how topic size tracks with other metrics for 20-topic model")
k=20
doc_lengths = LdaOutput.get_doc_lengths(corpus)
word_counts = LdaOutput.get_topic_wordcounts(theta_dict[k], doc_lengths, normalized = False)

LdaOutputTopicSizePlots.metric_size_comparison_grid(model = model_dict[k], 
                            corpus = corpus,
                            dictionary = dictionary,
                            sizes = word_counts,
                            label = "Expected Word Counts",
                            save_fig = True,
                            fig_outpath = plot_path,
                            fig_name = "20topic_metricsizeplot",
                            dpi = dpi,
                            fig_override = fig_override)



#also look at this for the 50-topic model
print("the same for 50-topic model")
k = 50
word_counts = LdaOutput.get_topic_wordcounts(theta_dict[k], doc_lengths, normalized = False)


LdaOutputTopicSizePlots.metric_size_comparison_grid(model = model_dict[k], 
                            corpus = corpus,
                            dictionary = dictionary,
                            sizes = word_counts,
                            label = "Expected Word Counts",
                            save_fig = True,
                            fig_outpath = plot_path,
                            fig_name = "50topic_metricsizeplot",
                            dpi = dpi,
                            fig_override = fig_override)




#HEAT MAPS FOR TOPIC SIMILARITY
print("\nCreating topic similarity heat maps")
LdaOutputTopicSimilarity.plot_topic_heatmap(pmodel = model_dict[25], qmodel = model_dict[20],
                      distance = "jensen_shannon",
                      normed = False,
                      save_fig = True, 
                      fig_outpath = plot_path,
                      fig_name = "2520TopicHeatmap",
                      dpi = dpi,
                      fig_override = fig_override)


LdaOutputTopicSimilarity.plot_topic_heatmap(pmodel = model_dict[20], qmodel = model_dict[15], 
                      distance = "jensen_shannon",
                      normed = False,
                      save_fig = True, 
                      fig_outpath = plot_path,
                      fig_name = "2015TopicHeatmap",
                      dpi = dpi,
                      fig_override = fig_override)



LdaOutputTopicSimilarity.plot_topic_heatmap(pmodel = model_dict[30], qmodel = model_dict[20], 
                      distance = "jensen_shannon",
                      normed = False,
                      save_fig = True, 
                      fig_outpath = plot_path,
                      fig_name = "3020TopicHeatmap",
                      dpi = dpi,
                      fig_override = fig_override)




# DOCUMENT TRAJECTORIES TO IDENTIFY SPLITTING - Example of documents from topic 15.3
print("Document trajectory plots to identify splitting dynamics")
LdaOutputDocs.plot_doc_trajectories(theta_dict = theta_dict,
                      root_k = 5,
                      topic_id = 3,
                      plot_all = False,
                      min_K = None,
                      max_K = None,
                      single_color = "purple",
                      figsize = (15,20),
                      shift_width = True,
                      save_fig = True,
                      fig_outpath = plot_path,
                      fig_name = "Demog5.3_doc_trajectories",
                      dpi = dpi,
                      fig_override = fig_override)
               

LdaOutputDocs.plot_doc_trajectories(theta_dict = theta_dict,
                      root_k = 15,
                      topic_id = 13,
                      plot_all = False,
                      min_K = None,
                      max_K = None,
                      single_color = "purple",
                      figsize = (15,20),
                      shift_width = True,
                      save_fig = True,
                      fig_outpath = plot_path,
                      fig_name = "Demog15.13_doc_trajectories",
                      dpi = dpi,
                      fig_override = fig_override)
               


# CORRELATION PLOT
print("Creating correlation plot")
_ = LdaOutputTopicSimilarity.topic_correlations(theta_dict[20], 
                                                plot = True,
                                                save_fig = True, 
                                                fig_outpath = plot_path,
                                                fig_name = "20_corrheatmap",
                                                dpi = dpi,
                                                fig_override = fig_override)




print("Overall topic over time grid")
LdaOutputTimePlots.plot_topic_sizes_by_year(theta_mat = theta_dict[20],
                          year_labels = year_labels,
                          sizetype = "word_count",
                          doc_lengths = doc_lengths,
                          plot_all_topics = True,
                          custom_topic_list = None,
                          right_zoom = 1, #don't plot 2021
                          save_fig = True,
                          fig_outpath = plot_path,
                          dpi = dpi,
                          fig_override = fig_override,
                          fig_name = "20TopicOverTime"
                          )
plt.close('all') #close to avoid consuming too much memory by leaving all these plots open






##### Comparing topic over time trends
print("Topic over time trends - comparison for 20, 25, 30 topic models")
theta_mat_list = [theta_dict[25], theta_dict[30]]
LdaOutputTimePlots.topic_size_by_year_model_comparison(year_labels = year_labels,
                                          main_model = model_dict[20], 
                                          theta_mat_main = theta_dict[20],
                                          model_list = [model_dict[25], model_dict[30]], 
                                          theta_mat_list = theta_mat_list,
                                          sizetype = "word_count", 
                                          doc_lengths = doc_lengths,
                                          plot_all_topics = True,
                                          custom_list = None,
                                          right_zoom = 1, #don't plot 2021
                                          save_fig = True,
                                          fig_name =  "20_FullComparison",
                                          fig_outpath = plot_path,
                                          dpi = dpi,
                                          fig_override = fig_override                                          
                                            )

#a few single plots
LdaOutputTimePlots.topic_size_by_year_model_comparison(year_labels = year_labels,
                                          main_model = model_dict[20],
                                          theta_mat_main = theta_dict[20],
                                          model_list = [model_dict[25], model_dict[30]], 
                                          theta_mat_list = theta_mat_list,
                                          sizetype = "word_count", 
                                          doc_lengths = doc_lengths,
                                          plot_all_topics = False,
                                          custom_list = [13],
                                          right_zoom = 1, #don't plot 2021
                                          save_fig = True,
                                          fig_name = "20_ComparisonPlot_13",
                                          fig_outpath = plot_path,
                                          dpi = dpi,
                                          fig_override = fig_override                                          
                                            )

LdaOutputTimePlots.topic_size_by_year_model_comparison(year_labels = year_labels,
                                          main_model = model_dict[20],
                                          theta_mat_main = theta_dict[20],
                                          model_list = [model_dict[25], model_dict[30]], 
                                          theta_mat_list = theta_mat_list,
                                          sizetype = "word_count", 
                                          doc_lengths = doc_lengths,
                                          plot_all_topics = False,
                                          custom_list = [1, 5, 8],
                                          right_zoom = 1, #don't plot 2021
                                          save_fig = True,
                                          fig_name = "20_ComparisonPlot_158",
                                          fig_outpath = plot_path,
                                          dpi = dpi,
                                          fig_override = fig_override                                          
                                            )


plt.close('all') #close to avoid consuming too much memory by leaving all these plots open
  




# TOPICS OVER TIME FOR 20-TOPIC MODEL
print("over time plot + bar plot pairs for 20-topic and 25-topic model")
per_year_size_dict = LdaOutput.get_per_group_topic_size(theta_dict[20], 
                                    label_list = year_labels,
                                    sizetype = "word_count", 
                                    doc_lengths = doc_lengths,
                                    normalized = True)


#FOR MODEL 20
for t in list(range(20)):
    print("20-Topic Model plot: %d" % t)
    LdaOutputTimePlots.plot_barplot_and_timeplot(model = model_dict[20], 
                                                  per_year_size_dict = per_year_size_dict,
                                                  custom_suptitle = "Topic 20.%d" % t,
                                                  theta_mat = theta_dict[20],
                                                  topic_id = t, 
                                                  sizelabel = "Expected word counts",
                                                  corpus = corpus, 
                                                  dictionary = dictionary, 
                                                  value_type = "counts",
                                                  right_zoom = 1, #don't include 2021
                                                  detect_max_vals = True,
                                                  figsize = (14,10),
                                                  topn = 20, 
                                                  lamb = 0.6,
                                                  save_fig = True, 
                                                  fig_outpath = os.path.join(wordplot_path, "20topic"), 
                                                  fig_name = "20_bartimeplot_%d"%t, 
                                                  dpi = dpi,
                                                  fig_override = fig_override)
                                                 
    plt.close('all') #close to avoid consuming too much memory by leaving all these plots open




#all Topics for model 25
per_year_size_dict = LdaOutput.get_per_group_topic_size(theta_dict[25], 
                                  label_list = year_labels,
                                  sizetype = "word_count", 
                                  doc_lengths = doc_lengths,
                                  normalized = True)

for t in list(range(25)):
    print("25-Topic Model plot: %d" % t)
    LdaOutputTimePlots.plot_barplot_and_timeplot(model = model_dict[25], 
                                                  per_year_size_dict = per_year_size_dict,
                                                  custom_suptitle = "Topic 25.%d" % t,
                                                  theta_mat = theta_dict[25],
                                                  topic_id = t, 
                                                  sizelabel = "Expected word counts",
                                                  corpus = corpus, 
                                                  dictionary = dictionary, 
                                                  value_type = "counts",
                                                  right_zoom = 1, #don't include 2021
                                                  detect_max_vals = True,
                                                  topn = 20, 
                                                  lamb = 0.6,
                                                  save_fig = True, 
                                                  fig_outpath = os.path.join(wordplot_path, "25topic"), 
                                                  fig_name = "25_bartimeplot_%d"%t, 
                                                  dpi = dpi,
                                                  fig_override = fig_override)
                                                 
    plt.close('all') #close to avoid consuming too much memory by leaving all these plots open




#HIV topics for model 30
per_year_size_dict = LdaOutput.get_per_group_topic_size(theta_dict[30], 
                                    label_list = year_labels,
                                    sizetype = "word_count", 
                                    doc_lengths = doc_lengths,
                                    normalized = True)


for t in [23, 29]:
    print("30-Topic Model plot: %d" % t)
    LdaOutputTimePlots.plot_barplot_and_timeplot(model = model_dict[30], 
                                                  per_year_size_dict = per_year_size_dict,
                                                  custom_suptitle = "Topic 30.%d" % t,
                                                  theta_mat = theta_dict[30],
                                                  topic_id = t, 
                                                  sizelabel = "Expected word counts",
                                                  corpus = corpus, 
                                                  dictionary = dictionary, 
                                                  value_type = "counts",
                                                  right_zoom = 1, #don't include 2021
                                                  detect_max_vals = True,
                                                  topn = 20, 
                                                  lamb = 0.6,
                                                  save_fig = True, 
                                                  fig_outpath = os.path.join(wordplot_path, "30topic"), 
                                                  fig_name = "30_bartimeplot_%d"%t, 
                                                  dpi = dpi,
                                                  fig_override = fig_override)
                                                  
    plt.close('all') #close to avoid consuming too much memory by leaving all these plots open







print("Examining how topics change over time")

#get topics by year
per_year_topics_dict20 = LdaOutput.get_per_group_topics(doc_wordtopic_arrays20, 
                                                      label_list = year_labels,
                                                      normalized = True,
                                                      save_dict = False) 

per_year_topics_dict25 = LdaOutput.get_per_group_topics(doc_wordtopic_arrays25, 
                                                      label_list = year_labels,
                                                      normalized = True,
                                                      save_dict = False) 

per_year_topics_dict30 = LdaOutput.get_per_group_topics(doc_wordtopic_arrays30, 
                                                      label_list = year_labels,
                                                      normalized = True,
                                                      save_dict = False) 


topn = 30
lamb = 0.6
size = (20,22)

print("20 topic model topics: words over time")
#PLOT ALL 20-TOPIC MODEL TOPICS
IDs_to_plot = list(range(20))     
for i in IDs_to_plot:                                             
    _ = LdaOutputTimePlots.plot_topic_by_year(per_year_topics_dict20, 
                        topic_id = i,
                        model = model_dict[20],
                        corpus = corpus,
                        dictionary = dictionary,
                        lamb = lamb, 
                        topn = topn,
                        left_zoom = 0,
                        right_zoom = 1, #don't plot 2021
                        color = "green", 
                        alpha = 0.25,
                        figsize = size,
                        save_fig = True,
                        fig_outpath = os.path.join(wordplot_path, "20topic"),
                        fig_name = "20Topics_%d_overtime" % i,
                        dpi = dpi,
                        fig_override = fig_override)                                                     
    
    plt.close('all') #close to avoid consuming too much memory by leaving all these plots open
  





#PLOT A FEW SPECIFC 25-TOPIC MODEL TOPICS
print("25 topic model topics")
IDs_to_plot = [23]     
for i in IDs_to_plot:                                             
    _ = LdaOutputTimePlots.plot_topic_by_year(per_year_topics_dict25, 
                        topic_id = i,
                        model = model_dict[25],
                        corpus = corpus,
                        dictionary = dictionary,
                        lamb = lamb, 
                        topn = topn,
                        left_zoom = 0,
                        right_zoom = 1,
                        color = "purple", 
                        alpha = 0.25,
                        figsize = size,
                        save_fig = True,
                        fig_outpath = os.path.join(wordplot_path, "25topic"),
                        fig_name = "25Topics_%d_overtime" % i,
                        dpi = dpi,
                        fig_override = fig_override)                                                     
    
    plt.close('all') #close to avoid consuming too much memory by leaving all these plots open

    
#PLOT A FEW SPECIFC 30-TOPIC MODEL TOPICS
print("30 topic model topics")
IDs_to_plot = [23, 29]     
for i in IDs_to_plot:                                             
    _ = LdaOutputTimePlots.plot_topic_by_year(per_year_topics_dict30, 
                        topic_id = i,
                        model = model_dict[30],
                        corpus = corpus,
                        dictionary = dictionary,
                        lamb = lamb, 
                        topn = topn,
                        left_zoom = 0,
                        right_zoom = 1,
                        color = "blue", 
                        alpha = 0.25,
                        figsize = size,
                        save_fig = True,
                        fig_outpath = os.path.join(wordplot_path, "30topic"),
                        fig_name = "30Topics_%d_overtime" % i,
                        dpi = dpi,
                        fig_override = fig_override)                                                     
    
    plt.close('all') #close to avoid consuming too much memory by leaving all these plots open
  

print("Tada!")



### 20.13 -- year-specific topics
LdaOutputGroupPlots.plot_topic_words_by_group(wordtopic_array = doc_wordtopic_arrays20,
                              group_list =  year_labels,
                              value_type = "counts",
                              custom_groups = list(range(1995,2021,5)),
                              to_plot_list = [13],
                              corpus = corpus,
                              dictionary = dictionary,
                              topn = 20,
                              lamb = 0.6, 
                              group_name = "Year",
                              plot_overall_magnitudes = True,
                              save_fig =  True,
                              fig_outpath = os.path.join(wordplot_path, "20topic"),
                              fig_name = "20.13_Year_Specific_Topics",
                              dpi = dpi,
                              fig_override = fig_override)
