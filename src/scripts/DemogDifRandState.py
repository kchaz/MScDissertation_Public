# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 17:36:24 2021


Train one LDA model using random state other than 175
Curious whether this breaks link between topic IDs when changing K

@author: kyla
"""


#standard imports
import sys
import pickle
import os
import logging

#gensim imports

# Import relative paths
from filepaths import code_path
from filepaths import demog_data_path as data_path
from filepaths import demog_models_randstate_path  as outpath

# Import functions
sys.path.insert(0, code_path)
import AbstractCleaner
import LdaLogging 
import LdaOutput
import LdaOutputWordPlots
from Helpers import file_override_request



# ask user to set whether want script to override existing files of same name or not
file_override = file_override_request()

#set dpi for saving figures
dpi = 150




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


##### PARAMETERS TO KEEP CONSTANT THROUGHOUT
eval_every = None  
per_word_topics = False  
chunksize = 1000
callback_list = None

num_topics = 25
alpha = "auto"
eta = 0.001
passes = 30
iterations = 200 

random_state = 502 #CHANGED


model = LdaLogging.LdaModelLogged(
                   log_filename = "random_state_test",
                   log_outpath = outpath,
                   resolve_name = True, 
                   raise_error = False,
                   save_model = True,      
                   model_fname = "random_state_test",
                   model_outpath = outpath,
                   callback_list = callback_list,
                   corpus = corpus,
                   id2word = dictionary,
                   chunksize = chunksize,
                   alpha = alpha,
                   eta = eta,
                   num_topics=num_topics,
                   eval_every = eval_every,
                   per_word_topics = per_word_topics,
                   random_state = random_state,
                   iterations = iterations, #changing each time
                   passes=passes)




theta_mat = LdaOutput.get_document_matrices(model = model,
                                               corpus = corpus,  
                                               minimum_probability = .01, #NOTE: somewhat arbitrary choice
                                               save_theta_matrix = True, 
                                               theta_outpath = outpath,
                                               theta_filename = "theta_mat_rex",
                                               per_word_topics = False,
                                               override = True                                      
                                          )


#create grid of topics

LdaOutputWordPlots.topic_relevance_grid(model = model,
                              corpus = corpus,
                              dictionary = dictionary,
                              value_type = "counts",
                              theta_mat = theta_mat, 
                              lamb = 0.6,
                              topn = 20,
                              minimum_probability = .01, 
                              first_title_only = False,
                              save_all_plots = True, 
                              dpi = dpi, 
                              fig_outpath = outpath,
                              fig_override = file_override,
                              display_figure = False
                              )





