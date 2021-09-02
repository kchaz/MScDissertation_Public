# -*- coding: utf-8 -*-
"""

This file runs a finalgrid search for demography data after deciding on eta
weights. Saves the best model for every K


using corpus formed from pre-processed abstracts with following
steps:
    
    1. basic cleaning
    2. ly word adjustment
    3. bigram merging
    4. removing words present in <5 documents


Author: Kyla Chasalow
Last edited: August 31, 2021


"""
import pickle
import sys
import logging 
import os

# Import relative paths
from filepaths import demog_data_path as data_path
from filepaths import code_path, demog_models_path

# Import functions
sys.path.insert(0, code_path)
import AbstractCleaner
import LdaGridSearch
import LdaLogging


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





# LDA MODEL PARAMETERS

#general set-up
eval_every = None 
per_word_topics = True # might want this   # late update: doesn't seem to make difference?
random_state = 175

#convergence parameters - checked for demography data via initial convergence analysis
passes = 30
iterations = 200
chunksize = 1000

#other important parameter(s)
alpha = "auto"


#Callbacks: only coherence
callback_list = ["coherence"]

#saving 
model_outpath = demog_models_path
log_outpath = demog_models_path




# LOAD OUTPUT OF INITIAL GRID SEARCH
GridOut = LdaGridSearch.load_GridEtaTopics("GridEtaTopics_output_initialrun", path = demog_models_path)


# RUN WEIGHTED GRID SEARCH
eta_weights = (.75,.25)

out_dict = LdaGridSearch.WeightedEtaTopicsSearch(GridOut, 
                                scalertype = "median_scaler",
                                aggregation_method = "median",
                                eta_weights = eta_weights,
                                save_best = False)


# GET list of tuples with (K, best_eta)
best_pairs = LdaGridSearch.get_best_parameter_pairs(out_dict)


# TRAIN AND SAVE EACH OF THESE MODELS
for pair in best_pairs:
    model = LdaLogging.LdaModelLogged(log_filename = "Best_%d" % pair[0], #format BestModel_K
                                   log_outpath = log_outpath,
                                   resolve_name = True, #resolve log name if already exists
                                   save_model = True,
                                   model_fname = "Best_%d" % pair[0],
                                   model_outpath = model_outpath,
                                   corpus = corpus,
                                   id2word = dictionary,
                                   chunksize = chunksize,
                                   alpha = alpha,
                                   eta = pair[1],
                                   num_topics = pair[0],
                                   iterations=iterations,
                                   passes = passes,
                                   eval_every = eval_every,
                                   per_word_topics = per_word_topics,
                                   random_state = random_state,
                                   callback_list = callback_list,
                                   topn_coherence = 10)
   

print("Tada! Your models have been trained")



