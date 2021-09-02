# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 20:02:50 2021

@author: kyla chasalow

"""
import os
import sys
import pickle

#gensim imports
from gensim.models.ldamodel import LdaModel


from filepaths import code_path
from filepaths import demog_data_path as data_path
from filepaths import  demog_models_path as model_path
from filepaths import demog_convergence_plots as plot_path

# Import functions
sys.path.insert(0, code_path)#r'C:\Users\kcsky\Documents\Oxford\STATISTICS\Dissertation\GitHub\MScDissertation\src')
import AbstractCleaner
import LdaOutputPerTopicMetrics
from Helpers import file_override_request



# ask user to set whether want script to override existing files of same name or not
file_override = file_override_request()

dpi = 200



# Load processed abstracts and get corpus and dictionary
filepath = os.path.join(data_path, "Demog_abstracts4_FreqFilter.pickle")
with open(filepath, "rb") as fp:   #Unpickling
    cleaned_docs = pickle.load(fp)
print(len(AbstractCleaner.extract_vocab(cleaned_docs)), "words in vocab")
print(AbstractCleaner.corpus_length(cleaned_docs), "words in corpus")
corpus, dictionary = AbstractCleaner.get_corpus_and_dictionary(cleaned_docs)




#LOAD ALL THE MODELS
Kvals = [5, 10, 15, 20, 25, 30 ,35, 40, 50, 60]

# LOAD ALL THE MODELS
fnames = ["Best_%d_model"%k for k in Kvals]
model_dict = {}
for i,k in enumerate(Kvals):
    model_dict[k] = LdaModel.load(os.path.join(model_path,fnames[i]))
    
model_list = [model_dict[k] for k in Kvals]
    
#examine sensitivity of rankings by mean coherence to choice of topn
K_labels = ["%d Topics" %k for k in Kvals]
_ = LdaOutputPerTopicMetrics.evaluate_coherence_sensitivity(
                                            model_list, 
                                            K_labels, 
                                            corpus = corpus, 
                                            dictionary = dictionary,
                                            title_annotation = "", 
                                            topn_range = (5,40,5),
                                            save_fig = True,
                                            fig_outpath = plot_path,
                                            fig_name = "sensivitity_analysis_varying_K",
                                            fig_override = file_override,
                                            dpi = dpi)


   

