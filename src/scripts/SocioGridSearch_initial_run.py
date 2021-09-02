# -*- coding: utf-8 -*-
"""

This file runs initial grid search for sociologydata. Meant to then be used
to analyze impact of using different weights for model selection. Does not
actually save any models


Author: Kyla Chasalow
Last edited: August 13, 2021


"""
import pickle
import sys
import logging 
import os

# Import relative paths
from filepaths import code_path
from filepaths import socio_models_path as model_path
from filepaths import socio_data_path as data_path
log_path = model_path


# Import functions
sys.path.insert(0, code_path)#r'C:\Users\kcsky\Documents\Oxford\STATISTICS\Dissertation\GitHub\MScDissertation\src')
import AbstractCleaner
import LdaGridSearch

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



# GRID SEARCH PARAMETERS

#general set-up
eval_every = None 
per_word_topics = False  #not saving models here so can be False
random_state = 175

#convergence parameters
passes = 30
iterations = 200
chunksize = 1000

#hyper parameters to search over
etas = [.001, .005, .01, .05, .1, .5, 1]
K =  [5, 10, 15, 20, 25, 30, 35, 40, 45] 
alpha = "auto"

#NO CALLBACKS - massively slows down code and makes file sizes larger
callback_list = None 




#Initial GRID SEARCH 

output = LdaGridSearch.GridEtaTopics(etas = etas, 
                                      num_topic_vals = K,
                                      log_progress = True,
                                      log_outpath = log_path,
                                      log_savenote = "_initialrun",
                                      callback_list = callback_list,
                                      corpus=corpus,
                                      dictionary=dictionary,
                                      chunksize = chunksize,
                                      passes = passes,
                                      iterations = iterations,
                                      eval_every = eval_every,
                                      random_state = random_state,
                                      per_word_topics = per_word_topics,
                                      topn_coherence = 10,
                                      topn_phf = 25,
                                      thresh = .01,
                                      alpha = alpha) 

# Save Grid Search Output
LdaGridSearch.save_GridEtaTopics(output, 
                                 filename = "GridEtaTopics_output_initialrun",
                                 outpath = model_path, 
                                 save_model_lists = False)


print("Tada!")
