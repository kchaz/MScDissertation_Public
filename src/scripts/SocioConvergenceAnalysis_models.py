# -*- coding: utf-8 -*-
"""

This file creates models for an initial analysis of convergence 
properties

Author: Kyla Chasalow
Last edited: August 10, 2021
"""

# SET THESE TO TRUE TO ACTUALLY RUN EACH EXPERIMENT(S)
RUN_EX1 = True
RUN_EX2 = True
RUN_EX3 = True
RUN_EX4 = True


#don't really need to save models here because mainly concerned with plots
#but can if you wish. Note this controls saving for all experiments
#if wish to just save from one experiment, can change save_model = True
#below for that particular experiment
SAVE_MODELS = False  




#standard imports
import sys
import pickle
import os
import logging

#gensim imports
from gensim.models.ldamodel import LdaModel

# Import relative paths
from filepaths import code_path
from filepaths import socio_data_path as data_path
from filepaths import  socio_models_convergence_path as model_path


# Import functions
sys.path.insert(0, code_path)
import LdaLogging 
import AbstractCleaner
import LdaOutput
import LdaOutputTopicSimilarity
from Helpers import file_override_request



# ask user to set whether want script to override existing files of same name or not
file_override = file_override_request()

#set dpi for saving figures
dpi = 150



# Load abstracts and apply last pre-processing step (removing words in under 5 documents)
filepath = os.path.join(data_path, "Socio_abstracts3_BigramAdder.pickle")
with open(filepath, "rb") as fp:   #Unpickling
    cleaned = pickle.load(fp)
    
cleaned5, corpus, dictionary, lens  = AbstractCleaner.filter_by_word_doc_freq(cleaned, no_below = 5, no_above = 1)
print("Initial Vocabulary size: ", lens[0])
print("Updated Vocabulary size: ", lens[1])

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


##### PARAMETERS TO KEEP CONSTANT THROUGHOUT
eval_every = None  
per_word_topics = False  
random_state = 175
chunksize = 1000
callback_list = ["perplexity", "coherence", "convergence"]






##### EXPERIMENT 1: eta = 0.1, K = 10, passes = 60, vary number of iterations


if RUN_EX1:
    print("Running Experiment 1")
     #key param
    num_topics = 10
    alpha = "auto"
    eta = 0.1
    passes = 60
    num_iter = [100, 150, 200, 250]
    
    #file names
    fnames = ["convergence_test_" + str(passes) + "_" + str(it) for it in num_iter]

    model_list = []
    for i, num in enumerate(num_iter):
        model_list.append(
              LdaLogging.LdaModelLogged(
                    log_filename = fnames[i],
                    log_outpath = model_path,
                    resolve_name = True, 
                    raise_error = False,
                    save_model = SAVE_MODELS,      
                    model_fname = fnames[i],
                    model_outpath = model_path,
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
                    iterations = num, #changing each time
                    passes=passes)
            )
        
        #Save summary array
        summary_array = LdaOutput.topic_summarizer(model_list = model_list, 
                           metric = "all",
                           corpus = corpus, 
                           dictionary = dictionary, 
                           topn_coherence=10, 
                           topn_phf= 25,
                           thresh = 0.01,
                           save_array = True, 
                           outpath = model_path,
                           filename = "convergence_test_summary_array",
                           override = file_override
                                          )





# EXPERIMENT 2: varying number of epochs trained so that can see what happens for training
# K = 10, eta = 0.1, passes = 200

if RUN_EX2:
    print("Running Experiment 2")
    #key parameters
    alpha = "auto"
    num_topics = 10
    iterations = 200
    eta = .1   
    
    num_passes = [0,1,2,10,20,30,40,50, 60]  #this is somewhat double - already did 60 above  
    
    fnames = ["convergence_test_" + str(p) + "_" + str(200) for p in num_passes]
    

    model_list = []
    for i, p in enumerate(num_passes):
        model_list.append(
              LdaLogging.LdaModelLogged(
                    log_filename = fnames[i],
                    log_outpath = model_path,
                    resolve_name = True, 
                    raise_error = False,
                    save_model = SAVE_MODELS,
                    model_fname = fnames[i],
                    model_outpath = model_path,
                    callback_list = callback_list,
                    corpus = corpus,
                    id2word = dictionary,
                    chunksize = chunksize,
                    alpha = alpha,
                    eta = eta, 
                    num_topics = num_topics,
                    eval_every = eval_every,
                    per_word_topics = per_word_topics,
                    random_state = random_state,
                    iterations = iterations,
                    passes=p) #changing each time
            )

    summary_array = LdaOutput.topic_summarizer(model_list = model_list, 
                           metric = "all",
                           corpus = corpus, 
                           dictionary = dictionary, 
                           topn_coherence=10, 
                           topn_phf= 25,
                           thresh = 0.01,
                           save_array = True, 
                           outpath = model_path,
                           filename = "epoch_test_summary_array",
                           override = file_override
                                          )


    #store topic difs
    LdaOutputTopicSimilarity.get_all_topic_difs(model_list, distance = "jensen_shannon",
                                                save_array = True,
                                                outpath = model_path,
                                                filename = "topic_dif_over_epoch_array",
                                                override = file_override)
    








##### DECISION: Setting passes = 40 since get rough convergence at least by then and 
##### for computational cost reduction
##### below two experiments are checks to make sure convergence results hold as vary
##### parameters. 


##### EXPERIMENT 3: eta = 0.1, passes = 40, iterations = 200, vary K
if RUN_EX3:
    print("Running Experiment 3")
    
    #check if already have the 10-topic model saved from above and if do, don't train it twice
    try:
        model_10 = LdaModel.load(os.path.join(model_path, "convergence_test_40_200_model"))
        num_topics = [5,25,50]
        already_exists = True
    except:
        num_topics = [5,10,25,50]
        already_exists = False
    

    #key parameters
    alpha = "auto"
    eta = 0.1
    passes = 40
    iterations = 200
    
    
    fnames = ["init_topic_test_" + str(k) for k in num_topics]  


    model_list = []
    for i, K in enumerate(num_topics):
        model_list.append(
              LdaLogging.LdaModelLogged(
                    log_filename = fnames[i],     #### LOGS GET SAVED
                    log_outpath = model_path,
                    resolve_name = True, 
                    raise_error = False,
                    save_model = SAVE_MODELS,       
                    model_fname = fnames[i],
                    model_outpath = model_path,
                    callback_list = callback_list,
                    corpus = corpus,
                    id2word = dictionary,
                    chunksize = chunksize,
                    alpha = alpha,
                    eta = eta,
                    num_topics = K, #changing each time
                    eval_every = eval_every,
                    per_word_topics = per_word_topics,
                    random_state = random_state,
                    iterations = iterations,
                    passes=passes)
            )


    if already_exists: #add in the 10 topic model if it was already trained and saved in experiment 1
        fnames = [fnames[0]] + ["convergence_test_40_200"] +  fnames[1:3]
        model_list = [model_list[0]] + [model_10] + model_list[1:3]
    
    
    summary_array = LdaOutput.topic_summarizer(model_list = model_list, 
                               metric = "all",
                               corpus = corpus, 
                               dictionary = dictionary, 
                               topn_coherence=10, 
                               topn_phf= 25,
                               thresh = 0.01,
                               save_array = True, 
                               outpath = model_path,
                               filename = "K_test_summary_array",
                               override = file_override
                                              )


    #store topic difs
    LdaOutputTopicSimilarity.get_all_topic_difs(model_list, distance = "jensen_shannon",
                                                save_array = True,
                                                outpath = model_path,
                                                filename = "topic_dif_by_K_array",
                                                override = file_override)
    


    
    






##### EXPERIMENT 4: K = 10, passes = 40, iterations = 200, vary eta


if RUN_EX4:
    print("Running Experiment 4")

    #key parameters
    alpha = "auto"
    num_topics = 10
    passes = 40
    iterations = 200
    
    etas = [.01, .1, 1, 10, 100]     
    fnames = ["init_eta_test_" + str(eta) for eta in etas]


    model_list = []
    for i, eta in enumerate(etas):
        model_list.append(
              LdaLogging.LdaModelLogged(
                    log_filename = fnames[i],
                    log_outpath = model_path,
                    resolve_name = True, 
                    raise_error = False,
                    save_model = SAVE_MODELS,
                    model_fname = fnames[i],
                    model_outpath = model_path,
                    callback_list = callback_list,
                    corpus = corpus,
                    id2word = dictionary,
                    chunksize = chunksize,
                    alpha = alpha,
                    eta = eta, #changing each time
                    num_topics = num_topics,
                    eval_every = eval_every,
                    per_word_topics = per_word_topics,
                    random_state = random_state,
                    iterations = iterations,
                    passes=passes)
            )


    summary_array = LdaOutput.topic_summarizer(model_list = model_list, 
                               metric = "all",
                               corpus = corpus, 
                               dictionary = dictionary, 
                               topn_coherence=10, 
                               topn_phf= 25,
                               thresh = 0.01,
                               save_array = True, 
                               outpath = model_path,
                               filename = "eta_test_summary_array",
                               override = file_override
                                              )


 



print("Done")