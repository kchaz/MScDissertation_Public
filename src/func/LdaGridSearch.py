# -*- coding: utf-8 -*-
"""

This file contains functions for running grid search to fit LDA topic models

Author: Kyla Chasalow
Last edited: September 1, 2021


"""
#BASIC FUNCTIONALITY
import numpy as np
import matplotlib.pyplot as plt
import pickle
import ast
import re
import os
import copy

#GENSIM
#from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel


#DEPENDENCIES
import LdaOutput
import LdaLogging
import Scatterbox

from Helpers import filename_resolver
from Helpers import figure_saver



   
    
   


################ Main Grid Search Functions

def GridEta(etas, num_topics, corpus=None, dictionary=None, alpha='auto', #primary parameters
                 log_progress = False, log_filename = "default", log_outpath = None, callback_list = None,  #logging parameters
                 resolve_name = True, raise_error = False,
                 distributed=False, chunksize=2000, passes=1, update_every=1, #rest of the LDA parameters
                 decay=0.5, offset=1.0, eval_every=10,
                 iterations=50, gamma_threshold=0.001, minimum_probability=0.01,
                 random_state=None, ns_conf=None, minimum_phi_value=0.01,
                 per_word_topics=False, dtype=np.float32, topn_coherence = 10,
                 topn_phf = 25, thresh = .01): 
    """
    
    Function to do a grid search over eta values for a gensim LDA model with
    a fixed number of topics. If log=True, creates a single log called filename.log to record
    progress for all models. 
    
    Does not actually identify best model direclty but provides a summary of each model 
    searched over (e.g. for use in WeightedEtaSearch)
    

    
    Logging can optionally use following three callbacks
        perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
        convergence_logger = ConvergenceMetric(logger='shell')
        coherence_umass_logger = CoherenceMetric(corpus=corpus, logger='shell', coherence = 'u_mass')
    But NOTE that this slows down the function significantly
    
    
    Does not return log file but creates it in working directory or log_outpath, writes to it, and closes it
    
    Parameters
    ----------
   
    etas : list of floats or ints
        contains eta values to grid search over    
    
    num_topics : int
        number of topics - held constant here for every model
     
    ------------log options---------------
    
    log_filename : str, optional, default = "default"
        name of logging file will be finename.log
        note: file name must not already exist or will raise error. If run file twice
        with default name "default" and log_progress = True, will raise this error
        
    log_outpath : str, optional, default None
        if specified, then log of name filename.log will be created
        (if no existing file has same name) in that directory. If
        outpath = None, simply creates it in current working directory  
        
    log_progress : bool, optional, default = False
        if True, will create a log as described above
      
        
    callback_list : list of strings or None, optional, default is None
    
        if None, logging will not track perplexity, coherence, or convergence over epochs
        otherwise, list can contain 1, 2 or all 3 of "coherence","convergence" and "perplexity"
        and will set up those callbacks for the log
        
        Note: if anything other than the three options above is given in list, function will simply
        ignore those
        
        WARNING: for reasons not entirely clear to me, using these here increases size of final model
        saved -- the worse the larger the grid of etas to search for. See more in docstring of log_setup()
        in LdaLogging.py. Using "convergence" logger here is particularly egregious and not advised.
        For this reason, default is not to use these here.
     
    -----------------------------------------------------
    resolve_name and raise_error:
        provide options for how to deal with the situation that proposed filename
        already exists (GridEta_K<savenote>.log) 
        
        1. if resolve_name is True, function will amend filename with _1, _2, _3 etc. until finds 
        one that does not already exist and then use that one. Will do this regardless
        of the value of raise_error
        
        2. if resolve_name is False, there is the possibility of trying to create a log 
        file that already exists. If proposed filename does not already exist, function
        simply creates that log. If it does already exist and raise_error = True, 
        function will raise error. If it does already exist and raise_error = False,
    ---------------------------
    
    topn_coherence : int, optional, default 10
        # of words to consider when calculating coherence
        
    topn_phf : int, optional, defualt = 25
        # of words to consider when calculating phf
        
    thresh : int in (0,1), optional
        threshold for phf. The default is 0.01.
        
                
    all other parameters are copied over from LdaModel with same defaults as in
    in gensim with exception of alpha, where default is here set to 'auto'
    all parameters from LdaModel included
    (though callbacks is here called callback_list and is not passed directly to
     LdaModel call)
    https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/ldamodel.py

        
    Returns
    -------
    
    dictionary with following keys:
    
    "etas" : the etas searched over
    
    "num_topics" : the fixed number of topics used
        
    "per_topic_summaries" contains a list of arrays where each entry corresponds to a model and contains a 4 x num_topics
        array where rows contain:
            0. coherences for each topic in model
            1. entropies for each topic in model
            2. KL divergences from overall corpus distribution for each topic in model
            3. phf (percent highest freqency, calibrated by topn_phf and thresh parameters above)
               for each topic in model
          
    "model_list"  : a list of gensim LDA model objects for each model trained
                                  
                                     
       
    """
    assert len(etas) > 0, "must give at least one eta value"
    assert num_topics > 0 and type(num_topics) == int, "num_topics must be a positive integer"
    assert all(np.array(etas) > 0), "All etas values must be positive"
    
    #to do: make it so it closes log if error happens
    
    #if asked to log, set-up logger, which checks if filename already exists
    callbacks = None #for if not logging
    if log_progress:
        log, file_handle, callbacks = LdaLogging.log_setup(filename = log_filename,
                              outpath = log_outpath,
                              corpus = corpus, 
                              topn = topn_coherence,
                              callback_list = callback_list,
                              resolve_name = resolve_name,
                              raise_error = raise_error)
        log.info('START: Optimizing %d-Topic Model' % (num_topics))
   
    
    #set-up holder for models
    model_list = [None]* len(etas)
    
    for i, eta in enumerate(etas):
        if log_progress:
            log.info('START: Training Model %d, eta=%f, num_topics=%d' % (i, eta, num_topics))
        
        model_list[i] = LdaModel(
            corpus = corpus,
            id2word = dictionary,
            alpha = alpha,
            eta = eta,
            num_topics=num_topics,
            passes=passes,
            chunksize = chunksize,
            iterations=iterations,
            eval_every = eval_every,
            update_every = update_every,
            per_word_topics = per_word_topics,
            random_state = random_state,
            callbacks = callbacks,
            decay = decay,
            offset = offset,
            gamma_threshold = gamma_threshold,
            minimum_probability = minimum_probability,
            minimum_phi_value = minimum_phi_value,
            ns_conf = ns_conf,
            dtype = dtype,
            distributed = distributed 
        )
        if log_progress:
            log.info('END: Training Model %d, eta=%f, num_topics=%d' % (i, eta, num_topics))
            log.info('--------------------------------------')
            log.info('--------------------------------------')
            
    #summary of per-topic metrics per model
    per_topic_summary_arrays = LdaOutput.topic_summarizer(model_list = model_list, 
                                                          metric = "all", 
                                                          corpus = corpus,
                                                          dictionary = dictionary, 
                                                          topn_coherence = topn_coherence,
                                                          topn_phf = topn_phf,
                                                          thresh = thresh)
       

    model_summaries = {}
    model_summaries["etas"] = etas
    model_summaries["num_topics"] = num_topics
    model_summaries["per_topic_summaries"] = per_topic_summary_arrays
    model_summaries["model_list"] = model_list #NEW
                                  
                                     
    
    #log summary of output, close log
    if log_progress:
        #write string version of output dictionary 
        log.info('--------------------------------------')
        log.info('--------------------------------------')
        log.info("SUMMARY: " + str(num_topics) + "-Topic Model, with grid over " + str(etas))
        log.info(str(model_summaries))   #NEW
        log.info('END SUMMARY:')
        log.info('END: Optimizing %d-Topic Model' % (num_topics))
        log.info('--------------------------------------')
        log.info('--------------------------------------')

        #close log at very end if opened it
        log.removeHandler(file_handle)
        del log, file_handle

    return(model_summaries)

          


def GridEtaTopics(etas, num_topic_vals, corpus = None, dictionary = None,  alpha = "auto", #primary parameters
                  log_progress = True, log_outpath = None, callback_list = None, log_savenote = "",    #logging parameters
                  raise_error = False, resolve_name = True, #what to do if file already exists
                  distributed=False, chunksize=2000, passes=1, update_every=1, 
                  decay=0.5, offset=1.0, eval_every=10,
                  iterations=50, gamma_threshold=0.001, minimum_probability=0.01,
                  random_state=None, ns_conf=None, minimum_phi_value=0.01,
                  per_word_topics=False, dtype=np.float32, topn_coherence = 10,
                  topn_phf = 25, thresh = .01):
    
    """
    
    Function to do a grid search over eta values and num_topics for a gensim LDA model. 
    If log_progress =True, creates a log for each K's grid search over etas 
    
    
    NOTE: asymmetric approach to grid-search here is motivated by observation that 
    may sometimes wish to consider multiple well-performing topic models with different
    numbers of topics. Thus this function is intended to allow user to either identify
    one best model of all options or to identify the best option for each number of topics
    using WeightedEtaTopicsSearch function below
    
    
    Logging  can optionally uses following three callbacks
        perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
        convergence_logger = ConvergenceMetric(logger='shell')
        coherence_umass_logger = CoherenceMetric(corpus=corpus, logger='shell', coherence = 'u_mass')
    But NOTE that this significantly slows down the function!
        
    
    Does not return log file but creates it in working directory, writes to it, and closes it
    
    
    Parameters
    ----------

    etas : list of floats or ints
        contains eta values to grid search over    
    
    num_topic_vals : list of ints
        contains number-of-topic values to grid search over
        
    ------------log options---------------
    
        
    log_progress : bool, optional, default = True
        if True, will create seprate logs for each K, for search over all eta
        logs will be named automatically: GridEta_K.log  
        
    log_outpath : str, optional, default None
        if specified, then logs from grid sarch will be created
        (if no existing file has same name) in that directory. If
        outpath = None, simply creates them in current working directory  
        
    log_savenote : str, optional, default ""  - optionally add on to file name 
              GridEta_K<savenote>.log     
        
    callback_list : list of strings or None, optional, default is None
    
        if None, logging will not track perplexity, coherence, or convergence over epochs
        otherwise, list can contain 1, 2 or all 3 of "coherence","convergence" and "perplexity"
        and will set up those callbacks for the log
        
        Note: if anything other than the three options above is given in list, function will simply
        ignore those
        
        WARNING: for reasons not entirely clear to me, using these here increases size of final model
        saved -- the worse the larger the grid of etas to search for. See more in docstring of log_setup()
        in LdaLogging.py. Using "convergence" logger here is particularly egregious and not advised.
        For this reason, default is not to use these here.
     
    -----------------------------------------------------
    resolve_name and raise_error:
        provide options for how to deal with the situation that proposed filename
        already exists (GridEta_K<savenote>.log) 
        
        1. if resolve_name is True, function will amend filename with _1, _2, _3 etc. until finds 
        one that does not already exist and then use that one. Will do this regardless
        of the value of raise_error
        
        2. if resolve_name is False, there is the possibility of trying to create a log 
        file that already exists. If proposed filename does not already exist, function
        simply creates that log. If it does already exist and raise_error = True, 
        function will raise error. If it does already exist and raise_error = False,
        then existing log file will be overridden
    -----------------------------------------------------
 
    topn_coherence : int, optional, default 10
        # of words to consider when calculating coherence
        
    topn_phf : int, optional, defualt = 25
        # of words to consider when calculating phf
        
    thresh : int in (0,1), optional
        threshold for phf. The default is 0.01.
        
    all other parameters are copied over from LdaModel with same defaults as in
    in gensim with exception of alpha, where default is here set to 'auto.'
    all parameters from LdaModel included
    (though callbacks is here called callback_list and is not passed directly to
     LdaModel call)
    https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/ldamodel.py

        
    Returns
    -------
    
    dictionary with keys equal to num_topic_vals and values equal to output from 
       GridEta (another dictionary) for that number of topics. 
       
    See GridEta() function for more details 
 
    """
    
    assert all(np.array(num_topic_vals) > 0), "All num_topic values must be positive"
    assert all(np.array(etas) > 0), "All etas values must be positive"

    model_dict = {}
    
    for K in num_topic_vals:
        
        log_filename = "GridEta_%d" % K + log_savenote
        
        #logs each individual model to seperate file
        model_dict[K] = GridEta(etas = etas, num_topics = K,  alpha=alpha, 
                                corpus=corpus, dictionary=dictionary, callback_list = callback_list,
                                log_progress= log_progress, log_outpath = log_outpath, log_filename = log_filename,
                                resolve_name = resolve_name, raise_error = raise_error, #amend name if already exists
                                distributed=distributed, chunksize=chunksize, passes=passes, 
                                update_every=update_every, decay=decay, offset=offset,
                                eval_every=eval_every, iterations=iterations,
                                gamma_threshold=gamma_threshold, minimum_probability=minimum_probability,
                                random_state=random_state, ns_conf=ns_conf, minimum_phi_value=minimum_phi_value,
                                per_word_topics=per_word_topics, dtype=dtype,
                                topn_coherence = topn_coherence, topn_phf = topn_phf, thresh = thresh)
      
    return(model_dict)





####### Functions for saving and loading output of grid search functions via pickle

def save_GridEta(output, filename, outpath = None, save_model_list = False,
                 resolve_name = True):
    """
    Checks whether filename.pickle already exists and if not,
    Uses pickle to save output from GridEta. 
    

    Parameters
    ----------
    output : dict
        output from grideta function
        
    filename : str
        desired filename for output file
        
    outpath : str, optional
        if specified, will save filename.pickle to this directory
        if None, saves it to working directory
        The default is None.
        
    save_model_list : Bool, optional
        if False, replaces model_list element of output with None so that
        actual models aren't saved (good for keeping file size smaller).
        The default is False and this is recommended.
        
     resolve_name : bool, optional
        if True, then if filename already exists in working directory (if outpath = None) or
        specified outpath, will not overwrite it. Instead, will add a _1 or, if that
        exists, _2 etc. to the end of the file.
        if False, will overwrite an existing file. Default is True

    Returns
    -------
    None.

    """
    output_checker(output, func = "GridEta")
    #checks
    assert type(filename) == str, "filename must be a string"
    if outpath is not None:
        assert type(outpath) == str, "outpath must be string if not None"

    if resolve_name:    
        #ammend filename if filename already exists
        filename = filename_resolver(filename = filename, 
                                             extension = "pickle", 
                                             outpath = outpath)                
    if not save_model_list:
        output = copy.deepcopy(output)
        output["model_list"] = None
        
    # incorporate outpath if given
    if outpath is not None:
        filepath = os.path.join(outpath, filename + ".pickle")
    else:
        filepath = filename + ".pickle"

    with open(filepath, "wb") as handle:   
        pickle.dump(output, handle) 



def save_GridEtaTopics(output, filename, outpath = None, save_model_lists = False,
                       resolve_name = True):
    """
    Uses pickle to save output from GridEtaOutput. 
    
    if save_model_lists = False, replaces "model_list" for each K
    with None and doesn't save LDA model objects in pickle. 
    
    Recommended not to save all the models -- only the best ones found via grid search
    (see Weighted grid search functions).

    Parameters
    ----------
    output : output dictionary from GridEtaTopics
        
    filename : name of file to use when saving, without filepath, which will be .pickle

    
    outpath : str, optional
        if specified, will save file here. Else, will save to working directory.
        The default is None.
  
    save_model_lists : bool, optional
        if False (recommended) replaced model_list for each K with None
        and thus doesn't save the actual LDA model objects in pickle,
        which would be expensive. The default is False.
  
    resolve_name : bool, optional
        if True, then if filename already exists in working directory (if outpath = None) or
        specified outpath, will not overwrite it. Instead, will add a _1 or, if that
        exists, _2 etc. to the end of the file.
        if False, will overwrite an existing file. Default is True

    Returns
    -------
    None.

    """
    output_checker(output, func = "GridEtaTopics")
    #checks
    assert type(filename) == str, "filename must be a string"
    if outpath is not None:
        assert type(outpath) == str, "outpath must be string if not None"

    if resolve_name:
        filename = filename_resolver(filename = filename, 
                                         extension = "pickle", 
                                         outpath = outpath)        
            
    #replace models with None
    if not save_model_lists:
        output = copy.deepcopy(output)
        for key in output.keys():
            output[key]["model_list"] = None
        
    # if doesn't already exist, create file, incorporate outpath if given
    if outpath is not None:
        filepath = os.path.join(outpath, filename + ".pickle")
    else:
        filepath = filename + ".pickle"

    with open(filepath, "wb") as handle:   
        pickle.dump(output, handle) 



def load_GridEta(filename, path = None):
    """Uses pickle to load output from from GridEta as saved by save_GridEta_output. 
    Optionally, specify file path to load it from.
    """
    if path is not None:
        filepath = os.path.join(path, filename + ".pickle")
    else:
        filepath = filename + ".pickle"
    with open(filepath, "rb") as handle:   
        out = pickle.load(handle) 
    return(out)



def load_GridEtaTopics(filename, path = None):
    """Uses pickle to load output from from GridEtaTopics as saved by save_GridEtaTopics_output. 
    Optionally, specify file path to load it from.
    
    NOTE: this is written just as wrapper for load_GridEta since the task is exactly the same
    (loading a pickle file). Added mainly for interpretability of having load/save pairs
    """
    return(load_GridEta(filename = filename, path = path))



  



###### MODEL SELECTION


# Given output of GridEta and/or GridEtaTopics, choose final model(s) using
# mean coherence, KL divergence, or some combination of the two
# mean coherence doesn't capture all desirable topic properties and sometimes
# leads to selection of model with topics that closely resemble overall corpus frequencies
# (high coherence, but not very specific, not useful), so KL divergence helps us
# select better models (when weighted appropriately - use visualize_weighted_.. 
# functions to explore consquences of different weights)


# TO DO: In long term,  might add other metrics (like entropy) as possibilities to be weighted



def _get_flat_values(output, metric):
    """
    output is GridEtaTopics output. Obtains the values of metric from every model
    run in the grid search (every K and eta combination) for purpose of 
    creating a scaler based on them
    
    * currently only set-up to handle coherence and kl-divergence *

    Parameters
    ----------
    output : dictionary, output from GridEtaTopics
    
    metric : str, either "coherence" or "kl" for now
    
    Returns
    -------
    a flattened array containing all the coherence values for all the topics
    in all the models trained in the grid search

    """
    output_checker(output, func = "GridEtaTopics")
    assert metric in ["coherence","kl"], "metric must be in ['coherence','kl']"
    if metric == "coherence":
        i = 0
    elif metric == "kl":
        i = 2
    
    Kvals = list(output.keys())
    #get all topic coherence values from each model in a flattened array
    all_vals = [np.array([elem[i] for elem in output[k]["per_topic_summaries"]]).flatten() for k in Kvals]
    #finish flattening
    all_vals = [elem for array in all_vals for elem in array] 
    return(all_vals)



class Scaler():
    """
    An instance contains all the summary statistics needed for each of
    three different kinds of scaling (methods below)
    
    Note: below "kl" corresponds to KL divergence of a topic from the overall corpus frequencies,
    as calculated in grid search
    
    Attributes:
    ----------
        self.coherence_vals   the array of coherence values used to fit scaler
        self.kl_vals          the array of kl values used to fit scaler
        self.min_co     minimum coherence value
        self.min_kl     minimum kl value
        self.max_co     maximum coherence value
        self.max_kl     maximum kl value
        self.mean_co    mean coherence value
        self.mean_kl    mean kl value
        self.std_co     standard deviation of coherence values
        self.std_kl     standard deviation of kl values
        self.median_co  median coherence value
        self.median_kl  median kl value
        self.iqr_co     inter-quartile range of the coherence values (Q3-Q1)
        self.iqr_kl     inter-quartile range of the kl values (Q3-Q1)
 
    Methods
    ----------
    minmax_scaler   -  scales all values to be between 0 and 1,  (x - min)/(max-min)
    mean_scalar     -  subtract mean and divide by std   (x - mean)/std  (sensitive to extremes)
    median_scalar   -  subtract median and divide by iqr  (x-median)/iqr  (less sensitive to extremes)
    
    """

    def __init__(self, coherence_vals, kl_vals):
        """

        Parameters
        ----------
        coherence_vals : 1-dimensional np.array containing coherence values
        kl_vals : 1-dimensional np.array containing kl divergence values

        """
        self.coherence_vals = coherence_vals
        self.kl_vals = kl_vals
        self.min_co = np.min(coherence_vals)
        self.min_kl = np.min(kl_vals)
        self.max_co = np.max(coherence_vals)
        self.max_kl = np.max(kl_vals)
        self.mean_co = np.mean(coherence_vals)
        self.mean_kl = np.mean(kl_vals)
        self.std_co = np.std(coherence_vals)
        self.std_kl = np.std(kl_vals)
        self.median_co = np.median(coherence_vals)
        self.median_kl = np.median(kl_vals)
        self.iqr_co = np.subtract(*np.percentile(coherence_vals, [75, 25]))
        self.iqr_kl = np.subtract(*np.percentile(kl_vals, [75, 25]))
    
    def minmax_scaler(self, vec, metric):
        assert type(vec) == np.ndarray, "vec must be a numpy array"
        assert metric in ["coherence","kl"], "metric must be 'coherence' or 'kl'"
        if metric == "coherence":
            return((vec - self.min_co)/(self.max_co - self.min_co))
        if metric == "kl":
            return((vec - self.min_kl)/(self.max_kl - self.min_kl))
        
    def mean_scaler(self, vec, metric):
        assert type(vec) == np.ndarray, "vec must be a numpy array"
        assert metric in ["coherence","kl"], "metric must be 'coherence' or 'kl'"
        if metric == "coherence":
            return((vec - self.mean_co)/self.std_co)
        if metric == "kl":
            return((vec - self.mean_kl)/self.std_kl)

    def median_scaler(self, vec, metric):
        assert type(vec) == np.ndarray, "vec must be a numpy array"
        assert metric in ["coherence","kl"], "metric must be 'coherence' or 'kl'"
        if metric == "coherence":
            return((vec - self.median_co)/self.iqr_co)
        if metric == "kl":
            return((vec - self.median_kl)/self.iqr_kl)
        
    # def test(self, vec, metric):
    #     assert type(vec) == np.ndarray, "vec must be a numpy array"
    #     assert metric in ["coherence","kl"], "metric must be 'coherence' or 'kl'"
    #     if metric == "coherence":
    #         return(vec/np.sum(self.coherence_vals))
    #     if metric == "kl":
    #         return(vec/np.sum(self.kl_vals))
    
    def __repr__(self):
        name_string = "Scaler object: Summary values for "
        name_string = name_string + " %d values each of topic coherence and topic KL-divergence from corpus:" % (len(self.coherence_vals))
        
        string_co = "Coherence Summary: min %f, max %f, mean %f, std %f, median %f, iqr %f" % (self.min_co,
                                                                                            self.max_co,
                                                                                            self.mean_co, 
                                                                                            self.std_co, 
                                                                                            self.median_co,
                                                                                            self.iqr_co)
        string_kl = "KL-Divergence Summary: min %f, max %f, mean %f, std %f, median %f, iqr %f" % (self.min_kl,
                                                                                            self.max_kl,
                                                                                            self.mean_kl, 
                                                                                            self.std_kl, 
                                                                                            self.median_kl,
                                                                                            self.iqr_kl) 
        return(name_string + "\n"*2 + string_co + "\n"*2 + string_kl)
        
    
    
    
def _get_combo_vals(coherences, kl_vals, weights, Scaler, scalertype, aggregation_method):
    """
    Helper function for use in WeightedEtaSearch and get_overall_best. 
    
    Takes lists of coherences and kl values corresponding to each model and
    
        0. scales them
        1. aggreagtes them
        2. applies linear combination to each one using weights
        
    Also handles various input checks 

    Parameters
    ----------
    coherences : list of lists of coherence values, each sublist representing a model

    kl_vals : list of lists of kl values, each sublist representing a model

    weights : tuple of length 2
        if do not sum to 1, function will normalize them to sum to 1
        thus (.5,.5) and (2,2) are equivalent
        
        element 0 corresponds to weight for coherence
        element 1 corresponds to weight for KL divergence
        
        
    Scaler : an instance of Scaler class 

    scalertype : str
        either "minmax_scaler", "mean_scaler", or "median_scaler"
        specifies which scaler to use
    
    aggregation_method : str
        either "mean" or "median", specifies whether to use average metric values
        or median metric values in weighted combination
        

    Returns
    -------
    a list of floats of same length as coherences and kl_vals lists
    contains the linear combination of aggregate scaled coherence and scaled kl values
    for each model represented in those lists

    """
    #various checks
    assert len(coherences) == len(kl_vals), "coherences and kl_vals must have same length"
    assert type(weights) == tuple and len(weights) == 2, "weights must be a length 2 tuple (#,#)"
    assert weights[0] >= 0 and weights[1] >= 0, "weights must be positive"
    assert aggregation_method in ["mean","median"], "aggregation method must be 'mean' or 'median'"
    assert scalertype in ["minmax_scaler",
                      "mean_scaler",
                      "median_scaler"], "scaletype must be one of ['minmax_scaler','mean_scaler','median_scaler']"

    #normalize weights if not already normalized
    if weights[0]+weights[1] != 1: 
        total = weights[0] + weights[1]
        weights = (weights[0]/total, weights[1]/total)
    
    
    if scalertype == "minmax_scaler":
        scaled_co = [Scaler.minmax_scaler(vec = elem, metric = "coherence") for elem in coherences]
        scaled_kl = [Scaler.minmax_scaler(vec = elem, metric = "kl") for elem in kl_vals]

    elif scalertype == "mean_scaler":  
        scaled_co = [Scaler.mean_scaler(vec = elem, metric = "coherence") for elem in coherences]
        scaled_kl = [Scaler.mean_scaler(vec = elem, metric = "kl") for elem in kl_vals]

    elif scalertype == "median_scaler": 
        scaled_co = [Scaler.median_scaler(vec = elem, metric = "coherence") for elem in coherences]
        scaled_kl = [Scaler.median_scaler(vec = elem, metric = "kl") for elem in kl_vals]

    #apply aggregation method to each
    if aggregation_method == "mean":
        agg_co = np.array([np.mean(elem) for elem in scaled_co])
        agg_kl = np.array([np.mean(elem) for elem in scaled_kl])

    elif aggregation_method == "median":
        agg_co = np.array([np.median(elem) for elem in scaled_co])
        agg_kl = np.array([np.median(elem) for elem in scaled_kl])

    #apply combination using weights
    combo_vals = (weights[0] * agg_co) + (weights[1] * agg_kl)
    return(combo_vals)
        






def WeightedEtaSearch(output, Scaler, scalertype, aggregation_method, weights,
                      model_outpath = None, save_best = False, 
                      savenote = "", save_override = False): #model saving parameter
    """
    
    
    The motivation for this function is that we might want to balance multiple 
    metrics when searching for best topic model. Currently, this function is 
    implemented to only look at a linear combination coherence and KL divergence 
    (divergence from overall corpus frequencies).
    
    High values of coherence and KL are desirable, but a preliminary analysis found they sometimes trade-off
    The function allows the user to specify weights for how much they want to consider 
    coherence versus KL divergence. The user can also specify one of three methods of
    scaling the metrics and can choose to aggregate them either by averaging or taking median
    
    
    Steps
    0. collect the eta values that were grid searched over and
       collect all the per-topic coherences and KL's in a list of arrays, one for each eta value
       
    1. scale all coherence entries and all KL entries using one of the  Scaler 
        options:
            "minmax_scaler" :   (x - min)/(max-min)
            "mean_scaler" :  (x-mean)/std
            "median_scaler" : (x-median)/iqr
            
        * note that reason for using Scaler in this way (supplying it externally)
        is that if we are looking at GridEtaTopics output, scaling must be using the ENTIRE
        collection of coherence and KL values, not just for particular search over etas for a
        given K but for all K and eta combos. 

    
    2. calculate mean or median (depending on aggregation_method argument) of topic metrics
       for each eta value
       
    3. calculate a weighted combination of the aggregated coherences and aaggregated kl
    
          (w1 * C) + (w2 * KL)
          
    4. "best" model is one with highest value of combo
    

    Example: for scalertype = "median_scaler", aggregation_method = "median",
        and weights = (.3, .7), function will return best model summary by
            0. scale coherences and KL divergence using median_scaler (see
               Scaler class definition for details)
            1. for each model, calculate the median scaled coherence and the median
                scaled kl divergence
            2. for each model, calculate  .3(scaled coherence) + .7 (KL divergence)
            
            Best model is then the model with maximum value in step 2.
    

    Parameters
    ----------
    output : output of GridEta function
 
    Scaler : an instance of Scaler class 

    scalertype : str
        either "minmax_scaler", "mean_scaler", or "median_scaler"
        specifies which scaler to use
    
    aggregation_method : str
        either "mean" or "median", specifies whether to use average metric values
        or median metric values in weighted combination
        
    weights : tuple of length 2
        if do not sum to 1, function will normalize them to sum to 1
        thus (.5,.5) and (2,2) are equivalent
        
        element 0 corresponds to weight for coherence
        element 1 corresponds to weight for KL divergence
        
        these weights are used to select a best model 
        
        
    -------------model saving options ------------------
    
    **NOTE: model files are automatically saved with format: GridEta_<num_topic>_model
    
    model_outpath : str, optional, default None
        if specified, then model files will be saved to that directory 
        If outpath = None, saves it in current working directory  

    save_best: bool, optional, default = False
        if True, will save the best model. 
    
    savenote: str, optional, default = ""
        optionally add something on to end of filename used to save model - GridEta_<num_topic><savenote>
        
    save_override : bool, optional, default = False
        if save_best = False, this does nothing
        if save_best = True, then if save_override = True, will save model while 
        overriding any existing models with same name. if save_override = False,
        will not do so and will instead adjust file name (adding _1, _2 etc. as necessary)
        until reaches a unique one 
       
    NOTE: If output["model_list"] is none (which can happen if output was 
           reconstructed from a log), then it will not throw an error but 
           simply will not save and will print a warning
       
    -----------------------------------------------------


    Returns
    -------
    Dictionary with following keys and values 
    
        "num_topics" : the number of topics
        "best_eta"  : the best eta value
        "best_combo_val"  : corresponding best value of linear combination
        "best_model_summary" : 4 x num_topics array with the 4 per-topic-metrics from LdaOutput.topic_summarizer()
        "best_model" : gensim LDA object for best model if available in output. If not, None
        "etas"      : the eta values used in the grid search
        "combination_vals"  : all the linear combiation values corresponding to each eta
        "scalertype"  : the type of scaling used (minmax, mean, median)
        "aggregation_method" : the type of aggregation method used (mean/median)
        "weights"  : the weights used        

    """
    output_checker(output, func = "GridEta")

    #collect values from output
    K = output["num_topics"]
    etas = output["etas"]
    coherences = [elem[0] for elem in output["per_topic_summaries"]]
    kl = [elem[2] for elem in output["per_topic_summaries"]]
    
    combos = _get_combo_vals(coherences = coherences,
                            kl_vals = kl,
                            weights = weights,
                            Scaler = Scaler,
                            scalertype = scalertype,
                            aggregation_method = aggregation_method)
    
    #find maximum value
    best_ind = np.argmax(combos)
    
    out_dict = {}
    out_dict["num_topics"] = K
    out_dict["best_eta"] = etas[best_ind]
    out_dict["best_combo_val"] = combos[best_ind]
    out_dict["best_model_summary"] = output["per_topic_summaries"][best_ind]
    
    if output["model_list"] is not None:
        out_dict["best_model"] = output["model_list"][best_ind]
    else:
        out_dict["best_model"] = None
        if save_best:
            save_best = False  #set save_best to False regardless of what it was 
            print("Warning: cannot save best model: model_list is None")
        
    out_dict["etas"] = etas
    out_dict["combination_vals"] = combos
    out_dict["scalertype"] = scalertype
    out_dict["aggregation_method"] = aggregation_method
    out_dict["weights"] = weights
        

    #navigate saving options
    if save_best:
        
        #best_model = output["model_list"][best_ind]
        proposed_name = "GridEta_%d_model" % (K) + savenote 
    
        if save_override: #don't look into whether file already exists - override if it does
                fname = proposed_name
        else: #extra protection so don't overwrite existing saved model
                fname = filename_resolver(proposed_name, 
                                           extension = None,
                                           outpath = model_outpath) 
     
        if model_outpath is not None:
                fname = os.path.join(model_outpath, fname)
     
        out_dict["best_model"].save(fname) 
        #best_model.save(fname)
    
    return(out_dict)
        

   


def WeightedEtaTopicsSearch(output, scalertype, aggregation_method, eta_weights,
                            save_best = False, savenote = "", save_override = False,
                            model_outpath = None): #model saving parameters
    """
    Finds best eta value for each number of topics included in a GridEtaTopics
    output dictionary using WeightedEtaSearch function. 
    
    Does not yet evaluate a best K -- see get_overall_best()
    function for this
    

    Parameters
    ----------
    output : dictionary as output by GridEtaTopics

    scalertype : str
        must be one of "minmax_scaler","mean_scaler", or "median_scaler"
        
    aggregation_method : str
        must be one of "mean" and "median"
        
    eta_weights : tuple of length 2
        first element is coherence weight
        second element is kl divergence weight
        
        called "eta weights" because used to select best eta for each K


    -------------model saving options ------------------
    
    **NOTE: model files are automatically saved with format: GridEta_<num_topic>_model
    
    model_outpath : str, optional, default None
        if specified, then model files will be saved to that directory 
        If outpath = None, saves it in current working directory  

    save_best: bool, optional, default = False
        if True, will save the best model for every K. 
    
    savenote: str, optional, default = ""
        optionally add something on to end of filename used to save model - GridEta_<num_topic><savenote>
        
    save_override : bool, optional, default = False
        if save_best = False, this does nothing
        if save_best = True, then if save_override = True, will save model while 
        overriding any existing models with same name. if save_override = False,
        will not do so and will instead adjust file name (adding _1, _2 etc. as necessary)
        until reaches a unique one 
       
    NOTE: If output["model_list"] is none (which can happen if output was 
           reconstructed from a log), then it will not throw an error but 
           simply will not save and will print a warning
       
    -----------------------------------------------------

    Returns
    -------
    0. dictionary with K values (number of topic values) as keys and 
    output of WeightedEtaSearch as values
    
    1. the instance of Scaler class used for scaling 
    

    """
    output_checker(output, func = "GridEtaTopics")
    #train scaler object on ALL the values that appear in output
    output_scaler = Scaler(_get_flat_values(output, metric = "coherence"),
                           _get_flat_values(output, metric = "kl"))
    
    #get best eta model for each K and put in a dictionary with K's as keys
    best_vals_dict = {k:WeightedEtaSearch(output[k],
                                   Scaler = output_scaler,
                                   scalertype = scalertype,
                                   aggregation_method = aggregation_method,
                                   weights = eta_weights,
                                   model_outpath = model_outpath,
                                   save_best = save_best,
                                   savenote = savenote,
                                   save_override = save_override)
                for k in output.keys()}
    
    
    return(best_vals_dict, output_scaler)




def get_best_parameter_pairs(output):
    """
    Gets just information about (K, best_eta) pairs from the output of
    WeightedEtaTopicsSearch. Returns a list of tuples where each tuple has 
    form  (K, best_eta)

    """
    Kvals = list(output[0].keys())
    return([(k,output[0][k]["best_eta"]) for k in Kvals])




def get_overall_best(output, K_weights = None, aggregation_method = None,
                     scalertype = None,
                     save_best = False, model_outpath = None, savenote = "",
                     save_override = False):
    """
    
    finds the best possible K value using possibly different weights from 
    those used to find the best eta values
   
    uses dictionary output by WeightedEtaTopicsSearch
    to obtain information about best model over eta values for each K
    and then applies a similar process to the one in WeightedEtaSearch
    to scale, aggregate, and calculate linear combination for each best-eta-model
    
    the final best K model returned is the one with best linear combo value
    over all the K
    
    Parameters
    ----------
    output : output of WeightedEtaTopicsSearch
    
    K_weights : tuple of length 2 containing ints or floats
        will be normalized if entries don't already sum to 1.
        Default is None. THese weights will be used to identify best K model.
        
        Note: if NONE, will just use the weights that were used to find the best eta values
        in WeightedEtaTopicsSearch        
    
    If K_weights is not None, then the following parameters must also be specified
    
        
        scalertype - "minmax_scaler","mean_scaler", or "median_scaler"
        
        aggregation_method - "mean" or "median"
        

    -------------model saving options ------------------
    
    **NOTE: if saved, model automatelly named:  GridEta_<num_topic>_model
    
    save_best: bool, optional, default = False
        if True, will save the best model 
    
    
    model_outpath : str, optional, default None
        if specified, then model will be saved to that directory 
        If outpath = None, saves it in current working directory  

    savenote: str, optional, default = ""
        optionally add something on to end of filename used to save model - GridEta_<num_topic><savenote>
        
    save_override : bool, optional, default = False
        if save_best = False, this does nothing
        if save_best = True, then if save_override = True, will save model while 
        overriding any existing model with same name. if save_override = False,
        will not do so and will instead adjust file name (adding _1, _2 etc. as necessary)
        until reaches a unique one 
       
    NOTE: If output["best_model"] is none (which can happen if output was 
           reconstructed from a log), then it will not throw an error but 
           simply will not save and will print a warning
       
        
    Returns
    -------
    dictionary summarizing best model as well as some info about the search.
    Keys are
    
    best_k : the K for the best model (best number of topics)
    best_eta : the eta value for the best model
    best_model_summary : 4xK array as output by LdaOutput.topic_summarizer() for best model
    best_model : best model object if available, else None
    K_weights : the weights used for the search
    combo_vals : the linear combination values for each model searched over
    all_model_rankings : the K values searched over, in order from best to worst in terms of combo values
    

    """
    output_checker(output, func = "WeightedEtaTopicsSearch")
    assert len(output) == 2, "output must be from WeightedEtaTopicsSearch, which has length 2"
    assert type(output[0]) == dict, "output must be from WeightedEtaTopicSearch, which has first entry a dictionary"
    #assert type(output[1]) == Scaler, "output must be from WeightedEatTopicSearch, which has second entry a Scaler object"
    
    Kvals = list(output[0].keys())
    
    if K_weights is None: #just use eta weights
        combos = [output[0][k]["best_combo_val"] for k in Kvals]
        best_ind = np.argmax(combos)
        best_k = Kvals[best_ind]
        weights = output[0][best_k]["weights"]
    
   
    else: 
        message = "If you specify K_weights, you must also specify scaler_type and aggregation_method"
        assert scalertype is not None and aggregation_method is not None, message
        best_model_summaries = [output[0][k]["best_model_summary"] for k in Kvals]
        coherences = [elem[0] for elem in best_model_summaries]
        kl = [elem[2] for elem in best_model_summaries]
        
        combos = _get_combo_vals(coherences = coherences,
                                 kl_vals = kl,
                                 weights = K_weights, 
                                 Scaler = output[1], 
                                 scalertype = scalertype,
                                 aggregation_method = aggregation_method)
    
        best_ind = np.argmax(combos)
        best_k = Kvals[best_ind]
        weights = K_weights
    
    best_eta = output[0][best_k]["best_eta"]
    best_model_summary = output[0][best_k]["best_model_summary"]

    
    out_dict = {}
    out_dict["best_K"] = best_k
    out_dict["best_eta"] = best_eta
    out_dict["best_model_summary"] = best_model_summary
    out_dict["best_model"] = output[0][best_k]["best_model"]
    out_dict["K_weights"] = weights
    out_dict["combo_vals"] = combos
    out_dict["all_model_rankings"] = np.array(Kvals)[np.flip(np.argsort(combos))]
    
    if out_dict["best_model"] == None:
        if save_best:
            save_best = False
            print("Warning: cannot save best model because best_model attribute is None")
    
    if save_best:
        proposed_name = "GridEta_%d_model" % (best_k) + savenote 
    
        if save_override: #don't look into whether file already exists - override if it does
                fname = proposed_name
        else: #extra protection so don't overwrite existing saved model
                fname = filename_resolver(proposed_name, 
                                           extension = None,
                                           outpath = model_outpath) 
     
        if model_outpath is not None:
                fname = os.path.join(model_outpath, fname)
     
        out_dict["best_model"].save(fname) 

    return(out_dict)





#---------------------------------------------------------------------------------

def visualize_weighted_eta(output, Scaler, scalertype, aggregation_method, num_weights = 10, zoom = 0,
                          figsize = (15,10), set_figsize = True, plot_legend = True, plot_title = True,
                          plot_short_title = False, top_text_rotation = 0, 
                          plot_annotation = True, annotation_color = "black",
                          plot_xlabel = True, plot_ylabel = True,
                          save_fig = False, fig_outpath = None, fig_name = None,
                          fig_override = False, dpi = 400):
    """
    
    Plot the values of weighted combination of coherence and KL as change 
    weights. 
    
    Some options are included to make this function usable for
    plotting in a grid (e.g. it is possible to remove the
                        individual plot titles)
    

    Parameters
    ----------
    output : output from GridEta
        
    Scaler : instance of class Scalar
    
    scalertype : str
        must be one of "minmax_scaler","mean_scaler", or "median_scaler"
        
    aggregation_method : str
        must be one of "mean" and "median"
        
    num_weights : int, optional
        The number of weights to examine. The default is 10.
        Too high a number here makes the graph very cluttered
        
    zoom : int, optional
        optionally, discard the zoom highest eta values. High eta values
        sometimes are so extreme that the rest of the eta lines get
        squished together into a single line. The default is 0.
        
    set_figsize : bool, optional
        control whether set figure size. Default is True.
        
    figsize : tuple, optional
        controls size of output figure. The default is (15,10).
        
        
    top_text_rotation: int, optional
        control whether annotations with best eta values at the top appear horitonal (0)
        or vertical (90). vertical recommended for condensed plots
        
    plot_annotation : bool, optional
        if true, annotates the top of the graph with the value of eta that attains
        maximum annotation value for each weight
        
    annotation_color: str, optional
        control color of top annotations with best eta values, default 'black'
        
    plot_xlabel, plot_ylabel , plot_tile, plot_legend are all bools that allo user
        to turn on or off their plot components. 
        
    plot_short_titl: bool, default False
        if true and plot_title is False, plots a shorter title. If
        plot_title is True, this does nothing
        
        
    saving options
    ---------------
    
    save_fig : bool, default False
        if True, saves figure
    
    outpath: str, default None
        if save_fig is True and this is None, saves to working directory
        else, saves to outpath
        
    fig_name : str, default None
         if save_fig is True, saves using this name. If None,
         automatically sets name to WeightedEtaViz.
         
         if fig_filename already exists at outpath, adds _1, _2, etc. 
         until finds iflename that does not already exist
         
    fig_override : bool, default is False
        if True, overrides any existing figures with same name

    dpi : int, default is 400
            dpi with which to save figures
        
    Returns
    -------
    0. a matrix of dimension  num_weights x # of etas. Each column corresponds to an 
    eta value and each row to a weight. Values are the linear combination of 
    coherence and KL as obtained via that weight
    
    1. a np.array containing the weights that are plotted in the graph
    
    2 a vector containing the best eta at each weight
    
    """
    output_checker(output, func = "GridEta")
    assert type(zoom) == int and zoom >= 0, "zoom must be non-negative integer"
    assert type(top_text_rotation) == int, "top_text_rotation must be an integer"
    
    #create a range of weights from 0 to 1 to examine
    co_weights = np.linspace(0,1,num_weights)
    kl_weights = 1 - co_weights
    
    #extract eta values and number of topics
    etas = output["etas"]
    K = output["num_topics"]
    
    #get combo values for each eta model for each pair of weights 
    summaries = [WeightedEtaSearch(output = output,
                                  Scaler = Scaler, 
                                  scalertype = scalertype, 
                                  aggregation_method = aggregation_method,
                                  weights = (w1,w2)) for  (w1, w2) in zip(co_weights, kl_weights)]

    #each column is for a given eta, each row is for a given weight
    per_eta_combo_vals = np.array([elem["combination_vals"] for elem in summaries])

    #identify max eta value at each weight for purpose of annotating graph
    eta_max = np.argmax(per_eta_combo_vals[:,:len(etas)-zoom], axis = 1) 
    
    #PLOTTING
    plt.rcParams.update({'font.family':'serif'})
    if set_figsize:
        plt.figure(figsize = figsize)
        
    #plot points and lines for each eta showing how linear combo value changes with weights
    for i in range(len(etas)-zoom):
        plt.scatter(co_weights, per_eta_combo_vals[:,i], label = "$\eta$ = " + str(etas[i]), s = 60)
        plt.plot(co_weights, per_eta_combo_vals[:,i])

    if plot_annotation:
        #annotations with which one is maximum since sometimes hard to see
        xmin, xmax, ymin, ymax = plt.axis()
        plt.text(co_weights[0] - .11, ymax*1.07, "Best $\eta$:", color  = annotation_color, fontsize = 12)
        for j in range(num_weights):
            plt.text(co_weights[j], ymax*1.07 , 
                     str(etas[eta_max[j]]), 
                     color = annotation_color, 
                     fontsize = 12,
                     rotation = top_text_rotation)   

    #plot aesthetics
    if plot_legend:
        plt.legend(fontsize = 14)
    if plot_xlabel:
        plt.xlabel("Coherence Weight ($w_1$) \n ($w_1 = 1 - w_2$)", fontsize = 16)
    if plot_ylabel:
        plt.ylabel("$w_1$(CO) + $w_2$(KL)",fontsize = 16)
        
    #title options, adjusting padding depending on whether have annotations
    if plot_annotation:
            pad = 60
    else:
            pad = 20
            
    if plot_title:
        title = "Grid Search Over $\eta$ values for K = %d " % K
        title = title + "using weighted linear combinations of \n %s scaled coherence " % (aggregation_method)
        title = title + "and %s scaled KL-divergence from corpus" % (aggregation_method)
        title = title + "\n (scaler = %s)" % scalertype
        plt.title(title,
                  pad = pad,
                  fontsize = 20)
    elif plot_short_title:
        plt.title("K = %d" % (K), 
                  pad = pad,
                  fontsize = 20)
        
        
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    #saving options
    if save_fig:
        if fig_name is None:
            fig_name = "WeightedEtaViz"
        figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     fig_override = fig_override,
                     dpi = dpi,
                     bbox_inches = "tight")
    
    best_etas = [etas[i] for i in eta_max]

    return(per_eta_combo_vals, co_weights, best_etas)



def visualize_weighted_eta_plots_grid(output, Kvals, aggregation_method, scalertype, annotation_color = "black",
                                      num_weights = 10, zoom = 0, top_text_rotation = 45, plot_suptitle = True,
                                      save_fig = False, fig_outpath = None, fig_name = None,
                                      fig_override = False, dpi = 400):
    """
    Plot a grid of up to four of the plots created by
    visualize_weighted_eta in a grid...more than that gets hard to see
    
    TO DO: Extend to allow different sizes of grids? at least allow 1, 2, or 3?

    Parameters
    ----------
    output : output from GridEtaTopics
    
    Kvals : list or numpy array of one to four K values to plot in grid
    
    scalertype : str
        must be one of "minmax_scaler","mean_scaler", or "median_scaler"
        
    aggregation_method : str
        must be one of "mean" and "median"
        
    num_weights : int, optional
        The number of weights to examine. The default is 10.
        Too high a number here makes the graph very cluttered
        
    zoom : int, optional
            optionally, discard the zoom highest eta values. High eta values
            sometimes are so extreme that the rest of the eta lines get
            squished together into a single line. The default is 0.
            
    top_text_rotation : int, optional
            controls rotation of best eta annotations at the top, 
            default is 45
            
    annotation_color : str, optional
            controls oclor of annotations at the top, default black
       
    plot_suptitle : bool, optional, default is True
        can optionally turn off title if desired
        
    saving options
    ---------------
    
    save_fig : bool, default False
        if True, saves figure
    
    outpath: str, default None
        if save_fig is True and this is None, saves to working directory
        else, saves to outpath
        
    fig_name : str, default None
         if save_fig is True, saves using this name. If None,
         automatically sets name to WeightedEtaVizGrid.
         
         if fig_filename already exists at outpath, adds _1, _2, etc. 
         until finds iflename that does not already exist
         
    fig_override : bool, default is False
        if True, overrides any existing figures with same name

    dpi : int, default is 400
            dpi with which to save figures
        

    Returns
    -------
    None.

    """
    output_checker(output, func = "GridEtaTopics")
    assert len(Kvals) <= 4, "I can handle at most 4 K values at a time"
    for k in Kvals:
        assert k in output.keys(), "%d is not in output keys" % k
    
    #adjust sizing depending on how many are in the plot
    if len(Kvals) == 4:
        nrows = 2
        ncols = 2
        figsize = (15,15)
        top = .80
    elif len(Kvals) == 3:
        nrows = 1
        ncols = 3
        figsize = (18, 8)
        top = .65   
    elif len(Kvals) == 2:
        nrows = 1
        ncols = 2
        figsize = (15,7)
        top = .65
    elif len(Kvals) == 1:
        nrows = 1
        ncols = 1
        figsize = (10,10)
        top = .70
        print("May I recommend using visualize_weighted_eta? This function is meant for more than one K")

    
    #train scaler on all coherence and kl values in output (including those potentially
    #not plotted here)
    output_scaler = Scaler(_get_flat_values(output, metric = "coherence"),
                           _get_flat_values(output, metric = "kl"))

    #build grid
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False, figsize= figsize)
    fig.text(0.5, -0.1, 'Coherence weight \n ($w_1 = 1 - w_2$)', ha='center', fontsize = 24)
    fig.text(-0.1, 0.5, r'$w_1$(CO) + $w_2$(KL)', va='center', rotation='vertical', fontsize = 24)
    fig.tight_layout(h_pad=15, w_pad = 10)
 
    #don't plot plot-specific x and y labels since all have same global label
    plot_xlabel = False
    plot_ylabel = False
    
    for i, k in enumerate(Kvals):
        plt.subplot(nrows,ncols,i+1)
        #only plot legend once to avoid clutter
        if i == 0:
            plot_legend = True
        else:
            plot_legend = False
    
        _ = visualize_weighted_eta(output[k], 
                                   Scaler = output_scaler,
                                   num_weights = num_weights,
                                   scalertype = scalertype,
                                   aggregation_method = aggregation_method, 
                                   set_figsize = False,
                                   plot_title = False,
                                   plot_annotation = True,
                                   annotation_color = annotation_color,
                                   plot_legend = plot_legend,
                                   plot_short_title = True,
                                   plot_ylabel = plot_ylabel,
                                   plot_xlabel = plot_xlabel,
                                   top_text_rotation = top_text_rotation,
                                   zoom = zoom,
                                   save_fig = False)
    if plot_suptitle:   
        suptitle = "Grid Search Over $\eta$ values for each K: varying weighted linear combination of \n"
        suptitle = suptitle + "topic coherence and topic kl-divergence from corpus"
        suptitle = suptitle + "\n (scaler: %s, aggregation method: %s )" % (scalertype, aggregation_method)
        plt.suptitle(suptitle, fontsize = 25)
        plt.subplots_adjust(top=top)     
    #saving options
    if save_fig:
        if fig_name is None:
            fig_name = "WeightedEtaVizGrid"
        figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     fig_override = fig_override,
                     dpi = dpi,
                     bbox_inches = "tight")
    
    
    




#---------------------------------------------------------------------------------


def _get_best_vals(output, Kvals):
    """
    
    helper function for use in visualize_weighted_eta_topics
    
    uses dictionary output by WeightedEtaTopicsSearch
    to obtain best linear combination values for each K in that 
    dictionary ** USING THE WEIGHTS THAT WERE USED FOR THE 
    SEARCH OVER ETA VALUES **
    
    Kvals is left as parameter because don't want to have to 
    re-calculate those every time in visualize_weighted_eta_topics.
    
    Function truly just meant as a helper below
    
    """
    return(np.array([output[0][k]["best_combo_val"] for k in Kvals]))



def _get_best_etas(output, Kvals):
    """
    
    helper function for use in visualize_weighted_eta_topics
    
    uses dictionary output by WeightedEtaTopicsSearch
    to obtain best eta values for each K in that 
    dictionary. 
    
    Kvals is left as parameter because don't want to have to 
    re-calculate those every time in visualize_weighted_eta_topics.
    
    Function truly just meant as a helper below

    """
    return(np.array([output[0][k]["best_eta"] for k in Kvals]))





def visualize_weighted_eta_topics(output, scalertype, aggregation_method,
                                  use_same_weights = True, eta_weights = None,
                                  num_weights = 10, zoom = 0,  figsize = (15,10), 
                                  set_figsize = True, plot_legend = True, plot_title = True,
                                  save_fig = False, fig_outpath = None, fig_name = None,
                                      fig_override = False, dpi = 400):
    """

    Parameters
    ----------
    output : output from GridEtaTopics
        
    scalertype : str
        must be one of "minmax_scaler","mean_scaler", or "median_scaler"
        
    aggregation_method : str
        must be one of "mean" and "median"
        
    num_weights : int, optional
        The number of weights to examine. The default is 10.
        Too high a number here makes the graph very cluttered
        
    use_same_weights : bool
        if True, uses the same weights used to calculate linear combination for 
        K to calculate linear combination for each eta and select a best eta for 
        each K. Thus each point along a line on the plot may correspond to a different
        (K, eta) pair because an optimal eta is selected for each K
        
        if False, eta_weights must be specified and then these weights will be used to 
        select a single eta for each K and that eta will be used throughout, even
        as the weights to select K vary along x axis of the plot
    
    eta_weights = 2-dimensional tuple 
        if weights do not sum to 1, they will be normalized to do so
        
        
    zoom : int, optional
        optionally, discard the zoom highest eta values. High eta values
        soemtimes are so extreme that the rest of the eta lines get
        squished together into a single line. The default is 0.
        
    set_figsize : bool, optional
        control whether set figure size. Default is True.
        
    figsize : tuple, optional
        controls size of output figure. The default is (15,10).
        
    plot_legend : bool, optional
        control whether plot legend or not, default True
        
    plot_title : bool, optional
        control whether plot title or not, default True
        
    saving options
    ---------------
    
    save_fig : bool, default False
        if True, saves figure
    
    outpath: str, default None
        if save_fig is True and this is None, saves to working directory
        else, saves to outpath
        
    fig_name : str, default None
         if save_fig is True, saves using this name. If None,
         automatically sets name to WeightedEtaTopicsViz.
         
         if fig_filename already exists at outpath, adds _1, _2, etc. 
         until finds iflename that does not already exist
         
    fig_override : bool, default is False
        if True, overrides any existing figures with same name

    dpi : int, default is 400
            dpi with which to save figures
        
    Returns
    -------
    
    a dictionary summarizing important values for the search over weights
    keys and values are:
        
        
    "Kvals" : the Kvalues searched over in the output given to the function
    
    "WeightxK_best_combo_vals" : a matrix of dimension  num_weights x # of K values. Each column corresponds to 
    K value and each row to a weight. Values are the best linear combination of 
    coherence and KL (over etas) as obtained via that weight for that K
    
    "WeightxK_best_eta_vals" : a matrix of dimension num_weights x # of K values. Each column corresponds to 
    K value and each row to a (coherence) weight. Values are the best eta values for that K
    when a given coherence weight is used to do the grid search
    
    "co_weights" : a np.array of length num_weights containing all the weights examined
    
    "per_weight_best_K_vals" : a np.array of length num_weights containing the best K at each weight
    

    """
    output_checker(output, func = "GridEtaTopics")
    co_weights = np.linspace(0,1,num_weights)
    kl_weights = 1 - co_weights
    
    #get K values contained in output
    Kvals = list(output.keys())
    assert zoom < len(Kvals), "zoom must be less than the number of K vals"
    
    #do grid search using each pair of weights to find best eta value
    if use_same_weights:
        weighted_search_outputs = [WeightedEtaTopicsSearch(output, 
                                    scalertype = scalertype,
                                    aggregation_method = aggregation_method,
                                    eta_weights = (w1,w2)) for (w1, w2) in zip(co_weights, kl_weights)]
        
        #get matrix of combo values each row a weight, each column corresponding to a K value
        #and each value corresponding to the best value of combo (for best eta) for that K
        #note that with this, we're getting combo value for best eta for each K and that best_eta
        #can change with weights, too
        per_K_combo_vals = np.array([_get_best_vals(elem, Kvals) for elem in weighted_search_outputs]) 
        
    
        #also matrix with the corresponding best eta values for each weight and K
        best_eta_vals = np.array([_get_best_etas(elem, Kvals) for elem in weighted_search_outputs])


    #do grid search once to find best eta for each K 
    #then use get_overall_best function to get combo values for each K while varying weights
    #but keeping best_eta constant for each K
    else:
        assert eta_weights is not None, "if use_same_weights = False, eta_weights must be specified"
        #do grid search once to find best eta for each K
        output = WeightedEtaTopicsSearch(output, 
                                                        scalertype = scalertype,
                                                        aggregation_method = aggregation_method,
                                                        eta_weights = eta_weights
                                                        )
        #get best eta value for each K (a vector)
        best_eta_vals = [output[0][k]["best_eta"] for k in Kvals] 
        
        
        weighted_search_outputs = [get_overall_best(output = output,
                                                    K_weights = (w1,w2),
                                                    scalertype = scalertype,
                                                    aggregation_method = aggregation_method,
                                                    save_best = False) for (w1,w2) in zip(co_weights, kl_weights)]
        
 
        #get matrix of combo values each row a weight, each column corresponding to a K value
        #and each value corresponding to the combo for that K (with best_eta constant from first search)
        per_K_combo_vals = np.array([elem["combo_vals"] for elem in weighted_search_outputs]) 
        
        
    
    
    
    #get best K for each weight for labeling purposes
    K_max = np.argmax(per_K_combo_vals[:,:len(Kvals)-zoom], axis = 1)
    best_K_vals = np.array([Kvals[i] for i in K_max])
      
   
    #PLOTTING
    plt.rcParams.update({'font.family':'serif'})
    if set_figsize:
            plt.figure(figsize = figsize)
        
    #labels for legend - if not using same weights for all, feasible to also note eta for each K
    labels = ["K = %d" % (Kvals[i]) for i in range(len(Kvals)-zoom)]
    if not use_same_weights:
        labels = [ elem + " ($\eta$ = " + str(best_eta_vals[i]) + ")" for i, elem in enumerate(labels)]
    
    
    #plot points and lines for each K showing how linear combo value changes with weights
    for i in range(len(Kvals)-zoom):
        plt.scatter(co_weights, per_K_combo_vals[:,i], label = labels[i], s = 60)
        plt.plot(co_weights, per_K_combo_vals[:,i])
    
    #annotations with which one is maximum since sometimes hard to see
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(co_weights[0] - .11, ymax*1.03, "Best K:", color  = "black", fontsize = 12)
    for j in range(num_weights):
        plt.text(co_weights[j], ymax*1.03 , str(Kvals[K_max[j]]), color = "black", fontsize = 12)   
    
        
    #legend 
    if plot_legend:
        plt.legend(fontsize = 14)
        

    plt.xlabel("Coherence Weight ($w_1$) \n ($w_1 = 1 - w_2$)", fontsize = 16)
    plt.ylabel("$w_1$(CO) + $w_2$(KL)",fontsize = 16)
    if plot_title:
        plt.title("Model Selection for K, for different choices of weights \n (scaler: %s, aggregation method: %s )" %(scalertype, aggregation_method),
                      pad = 45,
                      fontsize = 20)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    
    #note about eta values in case of use same weights - positioning here may be a bit temperamental / ad hoc
    if use_same_weights:
        message = "$\eta$ values not shown. Each point represents K-topic model with $\eta$ also optimized using given weight"
        if ymin <0:
            plt.text(0.05, ymin * 2.5,message, fontsize = 12)   
        elif ymin == 0: 
            plt.text(0.05, ymin - .3,message, fontsize = 12)  
        else:
            plt.text(0.05, ymin * .90,message, fontsize = 12)  
        
    #create summary dictionary
    out_dict = {}
    out_dict["Kvals"] = Kvals
    out_dict["best_combo_vals"] = per_K_combo_vals
    out_dict["best_eta_vals"] = best_eta_vals
    out_dict["co_weights"] = co_weights
    out_dict["per_weight_best_K_vals"] = best_K_vals
    
    
    if save_fig:
        if fig_name is None:
            fig_name = "WeightedEtaTopicViz"
        figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     fig_override = fig_override,
                     dpi = dpi,
                     bbox_inches = "tight")
    
    return(out_dict)
            
            

            

            
    
    
########## Functions for re-constructing output of the GridEta and GridEtaTopics functions
########## from log(s). Useful in case there is an error or have to turn off computer
########## for some reason and GridEtaTopics function doesn't finish running - can then still 
########## recover necessary information for models it did run even when didn't get to apply
########## save function to save output.

def _extract_GridEta_dict(s):
    """
    Helper for reconstructing GridEtaTopics output from log output
    
    s is a string version of a summary of a single GridEta output as entered in log
 

    output is a dictionary of form output by GridEta 

    the one difference is that "model_list" is set to None as this is not reoverable from log                          

    """
    
    #extract model summary info
    K = int(re.findall(r"num_topics': (\d+)",s)[0])
    etas = ast.literal_eval(re.findall(r"'etas': (.+), 'num_topics",s)[0])
    step1 = re.findall(r"per_topic_summaries': \[(.+)\], 'model_list",s)[0] #isolate the array bit
    step2 = step1.split("array")[1:] #split by "array" so get a list of summary arrays for model
    step3 = [   ast.literal_eval(re.findall(r"\((.*)\)",elem)[0])   for elem in step2] #get actual array
    per_topic_summaries = [np.array(elem) for elem in step3] #turn them into numpy arrays
    
    #form it into dictionary
    out_dict = {}
    out_dict["etas"] = etas
    out_dict["num_topics"] = K
    out_dict["per_topic_summaries"] = per_topic_summaries
    out_dict["model_list"]= None
    out_dict
    return(out_dict)


def reconstruct_GridEta_Output(logfilename, path = None):
    """
    
    reconstruct information output by GridEta from its log
    cannot recover models this way but can recover useful information 
    for plotting and analysis. 
    
    Intended as a backup to help re-construct output if GridEtaTopics does not
    run all the way
    

    Parameters
    ----------
    logfilename : name of a log as created by GridEtaTopics without the ".log" part
        assumes that for each K-topic model recorded in log, results for all
        eta grid search models are present

    path : str, optional, default None
        if path is specified, then looks for log file in that directory
        else, looks in working directory
        
 
    Returns
    -------
    0. a dictionary identical to the output of GridEta for that log, with place where best model object would
    be replaced by None

    1. the number of topics K that was used in GridEta function that produced log
    2. list containing the eta values searched over
    """

    #load log
    if path is not None:
        filepath = os.path.join(path, logfilename + ".log")
    else:
        filepath = logfilename + ".log"
    
    file = open(filepath, "r")
    log = file.read()
    
    #separate each line of log into an element of list
    log_list = log.split("\n")
    
    #extract number of topics
    Kvals = [int(k) for e in log_list for k in re.findall(r'(\d+)-Topic Model', e)] 
      
    assert len(np.unique(Kvals)) == 1, "This log appears to contain Topic models with different numbers of topics. Got %d different K values" % len(np.unique(Kvals))
    K = Kvals[0]
    
    #extract start and end indices of summary
    start_ind = [i for (i,e) in enumerate(log_list) if re.search(r': SUMMARY:', e)]  
    end_ind = [i for (i,e) in enumerate(log_list) if re.search(r': END SUMMARY:', e)]    
    
    assert len(start_ind) == 1 and len(end_ind) ==1, "This log appears to contain more than one summary - obtained multiple start or end indices"
    
    #extract etas searched over
    etas = ast.literal_eval(re.findall(r"grid over (.+)",log_list[start_ind[0]])[0])
    
    #isolate the summary and turn it into a single string again
    summary_list = log_list[(start_ind[0]+1):end_ind[0]]
    summary_string = "".join(summary_list)
    
    #remove initial date-time info from beginning
    dict_string = re.findall(r"INFO: (.+)", summary_string)[0]   
    
    out_dict = _extract_GridEta_dict(dict_string)


    file.close()
    return(out_dict, K, etas) 


def reconstruct_GridEtaTopics_Output(logfile_lst, path = None):
    """
    
    reconstruct output from GridEtaTopics using a list of log files for each 
    K

    Parameters
    ----------
    logfile_lst : list of strings
        should contain names of log files without the .log

    path : str, optional, default None
        if path is specified, then looks for all log files in that directory
        else, looks in working directory
        

    Returns
    -------
    output dictionary as would be output by GridEtaTopics if run over each of 
    K-values represented by log
    
    checks that there are no repeats in K values and that all have same eta values
    if not, throws error.

    """
    assert type(logfile_lst) == list, "logfile_lst must be a list"
    
    reconstructions = [reconstruct_GridEta_Output(file, path) for file in logfile_lst]
    Kvals = [elem[1] for elem in reconstructions]
    etas = [elem[2] for elem in reconstructions]

    #Check no K values included twice
    assert len(np.unique(Kvals)) == len(Kvals), "Some logs have the same K value"
    #Check all were grid searches over same etas
    assert all(elem == etas[0] for elem in etas), "Logs do not all have same eta grid"
    
    out_dict = {k:elem[0] for (k,elem) in zip(Kvals,reconstructions)}
    return(out_dict)





    
    
##################### FUNCTIONS FOR PLOTTING GRID SEARCH OUTPUT ##################


#Helper's to extract info from GridEta output for GridEta_scatterbox

def _extract_coherences(output):
    """Output is output from GridEta function. Extracts list of arrays where
    each array contains coherences for each model searched over in call to
    GridEta function"""
    return([elem[0] for elem in output["per_topic_summaries"]])

def _extract_entropies(output):
    """Output is output from GridEta function. Extracts list of arrays where
    each array contains entropies for each model searched over in call to
    GridEta function"""
    return([elem[1] for elem in output["per_topic_summaries"]])

def _extract_kl(output):
    """Output is output from GridEta function. Extracts list of arrays where
    each array contains KL_divergences from corpus for each model searched over in call to
    GridEta function"""
    return([elem[2] for elem in output["per_topic_summaries"]])

def _extract_phf(output):
    """Output is output from GridEta function. Extracts list of arrays where
    each array contains PHF values for each model searched over in call to
    GridEta function"""
    return([elem[3] for elem in output["per_topic_summaries"]])


def _extract_etas_as_strings(output):
    """Output is output from GridEta function. Extracts etas used in grid search
    and turns them into strings to be used for labelling """
    return([str(elem) for elem in output["etas"]])


def _extract_num_topics(output):
    """Output is output from GridEta function, returns the fixed number of
    topics used in the GridEta function"""
    return(len(output["per_topic_summaries"][0][0]))





def GridEta_scatterbox(output, metric, topn_coherence = None, topn_phf = None, thresh = None,
                       save_fig = False, fig_outpath = None, fig_name = None,
                                      fig_override = False, dpi = 400):
    """
    Plot a boxplot overlaid with points (a "ScatterBox") for each model considered
    in GridEta. Used to get more insight on the differences between the models
    considered there 
    
    (e.g. is the highest mean coherence model only selected because 
     the other models have a few extremely low coherence topics or is there a consistent shift?)
    
    Metrics available
    -----------------
        coherence:  'u_mass' coherence of each topic computed using topn_coherence words
        
        phf: stands for "percent high frequency." This is calculated via the
             percent_high_freq() function. It is the percentage of topn_phf
             words in each topic that come from the top (1-thresh)% of words
             by frequency in the overall corpus. Used to examine whether
             topics are mainly relying on high-frequency words
   
        kl: Kullback-Leibler divergence (relative entropy) between the topic 
            and the overall corpus. Larger values mean topic is more distinct
            from corpus-wide probabilities of each word. 0 means they are identical.

        entropy: the entropy (H = -sum_i p_i * ln(p_i)) of each topic. 
            
    
    Parameters
    ----------
    output : output from GridEta function, without any modifications 
        
    metric : str
        must be one of ['coherence','phf','kl','entropy','all']
       
    **note: topn_coherence, topn_phf, and thresh will have been set in call to GridEta that
      generated output so arguments here are just for plotting purposes!

        topn_coherence : int, optional
            # of words to considered when calculating coherence.
            Must be specified if metric = "all" or "coherence"
            but otherwise can leave default None
         
        topn_phf : int, optional
            # of words to consider when calculating phf 
            must be specified if metric = "all" or "phf"
            but otherwise can leave default None
          
        thresh : int in (0,1), optional
            threshold for phf.
            must be specified if metric = "all" or "phf"
            but otherwise can leave default None
       
    saving options
    ---------------
    
    save_fig : bool, default False
        if True, saves figure
    
    outpath: str, default None
        if save_fig is True and this is None, saves to working directory
        else, saves to outpath
        
    fig_name : str, default None
         if save_fig is True, saves using this name. If None,
         automatically sets name to GridEtaScatterbox.
         
         if fig_filename already exists at outpath, adds _1, _2, etc. 
         until finds iflename that does not already exist
         
    fig_override : bool, default is False
        if True, overrides any existing figures with same name

    dpi : int, default is 400
            dpi with which to save figures
        

    Returns
    -------
    Returns summary array for whatever metric selected 
    (e.g. for coherence, a list of arrays with each array containing coherences
     for each model)
    
    In case of metric = "all" returns list of lists containing these
    [coherences, entropies, kl, phf]
    
    Plots
    -------
    if metric is not "all" it plots a scatterbox of the metric given, comparing
    its spread for each model examined in GridEta search
    
    if metric is "all" it plots all four metrics in a 4x4 grid

    """
    output_checker(output, func = "GridEta")
    assert list(output.keys()) == ['etas', 'num_topics', 'per_topic_summaries', 'model_list'], "output must be from GridEta function. Keys don't match"
    assert metric in ['coherence','phf','kl','entropy','all'], "metric must be one of ['coherence','phf','kl','entropy','all']"
    
    
    
    num_topics = _extract_num_topics(output)
    etas =  _extract_etas_as_strings(output)
    
    if metric == "coherence":
        assert topn_coherence is not None, "topn_coherence must be specified for coherence"
             
        coherences = _extract_coherences(output)
        title = "Spread of Coherence for Topics in each Model (K = %d)" % num_topics
        title += "\n(using top %d words)" % topn_coherence
        Scatterbox.scatterbox(arr_list = coherences, labels = etas,
                              plot_points = True,
                              xlabel = r"$\eta$", ylabel = "Coherence",
                              title = title,
                              color_box = "purple", color_point = "blue",
                              alpha_point = .5, alpha_box = .3, legend_point_label= "Topics")
        if save_fig:
            if fig_name is None:
                fig_name = "GridEtaScatterbox"
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         fig_override = fig_override,
                         dpi = dpi,
                         bbox_inches = "tight")
        
        return(coherences)
    
    elif metric == "entropy":
        entropies = _extract_entropies(output)
        Scatterbox.scatterbox(arr_list = entropies, labels = etas,
                              plot_points = True,
                              xlabel = r"$\eta$", ylabel = "Entropy",
                              title = "Spread of Entropy for Topics in each Model (K = %d)" % num_topics,
                              color_box = "lightgreen", color_point = "blue",
                              alpha_point = .5, alpha_box = .4, legend_point_label= "Topics")
        if save_fig:
            if fig_name is None:
                fig_name = "GridEtaScatterbox"
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         fig_override = fig_override,
                         dpi = dpi,
                         bbox_inches = "tight")
        
        
        return(entropies)
    
    elif metric == "kl":
        kl = _extract_kl(output)
        Scatterbox.scatterbox(arr_list = kl, labels = etas,
                              plot_points = True,
                              xlabel = r"$\eta$", ylabel = "KL-Divergence from Corpus Distribution",
                              title = "Spread of KL-Divergence for Topics in each Model (K = %d)" % num_topics,
                              color_box = "lightblue", color_point = "blue",
                              alpha_point = .5, alpha_box = .4, legend_point_label= "Topics")
        if save_fig:
            if fig_name is None:
                fig_name = "GridEtaScatterbox"
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         fig_override = fig_override,
                         dpi = dpi,
                         bbox_inches = "tight")
        
        return(kl)
    
    elif metric == "phf":
        assert thresh is not None, "thresh must be specified for phf"
        assert topn_phf is not None, "topn_phf must be specified for phf"
            
        phf = _extract_phf(output)
        title = "Spread of PHF for Topics in each Model (K = %d)" % num_topics
        title += "\n(using top %d words and %s threshold)" % (topn_phf, str(thresh))
        Scatterbox.scatterbox(arr_list = phf, labels = etas,
                              plot_points = True,
                              xlabel = r"$\eta$", ylabel = "Percentage High Frequency",
                              title = title,
                              color_box = "orange", color_point = "blue", ylim = (-.05,1.05),
                              alpha_point = .5, alpha_box = .3, legend_point_label= "Topics")
        if save_fig:
            if fig_name is None:
                fig_name = "GridEtaScatterbox"
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         fig_override = fig_override,
                         dpi = dpi,
                         bbox_inches = "tight")
        
        return(phf)
    
    
    
    #----------------------------------all----------------------------------
    else:
        assert thresh is not None, "thresh must be specified for when plotting all metrics"
        assert topn_phf is not None, "topn_phf must be specified when plotting all metrics"
        assert topn_coherence is not None, "topn_coherence must be specified when plotting all metrics"
             
        
        #get all the values
        coherences = _extract_coherences(output)
        entropies = _extract_entropies(output)
        kl = _extract_kl(output)
        phf = _extract_phf(output)
        
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(20, 20))
        fig.text(0.5, -0.06, '$\eta$ values', ha='center', fontsize = 24)
        fig.text(-0.06, 0.45, 'Per-Topic Metric', va='center', rotation='vertical', fontsize = 24)
        fig.tight_layout(h_pad=14, w_pad = 6)
        
        
        plt.subplot(2,2,1)
        title = "Spread of Coherence for Topics in each Model"
        title += "\n(using top %d words)" % topn_coherence
        Scatterbox.scatterbox(arr_list = coherences, labels = etas,
                              plot_points = True,
                              xlabel = r"$\eta$", ylabel = "Coherence",
                              title = title,
                              color_box = "purple", color_point = "blue",
                              alpha_point = .5, alpha_box = .3, set_fig_size = False,
                              plot_legend = False)
        
        plt.subplot(2,2,2)
        Scatterbox.scatterbox(arr_list = entropies, labels = etas,
                              plot_points = True,
                              xlabel = r"$\eta$", ylabel = "Entropy",
                              title = "Spread of Entropy for Topics in each Model",
                              color_box = "lightgreen", color_point = "blue",
                              alpha_point = .5, alpha_box = .4, set_fig_size = False,
                              plot_legend = True, legend_point_label= "Topics")
  
        plt.subplot(2,2,3)
        Scatterbox.scatterbox(arr_list = kl, labels = etas,
                              plot_points = True,
                              xlabel = r"$\eta$", ylabel = "KL-Divergence from Corpus Distribution",
                              title = "Spread of KL-Divergence for Topics in each Model",
                              color_box = "lightblue", color_point = "blue",
                              alpha_point = .5, alpha_box = .4, set_fig_size = False,
                              plot_legend = False)

        plt.subplot(2,2,4)
        title = "Spread of PHF for Topics in each Model"
        title += "\n(using top %d words and %s threshold)" % (topn_phf, str(thresh))
        Scatterbox.scatterbox(arr_list = phf, labels = etas,
                              plot_points = True,
                              xlabel = r"$\eta$", ylabel = "Percentage High Frequency",
                              title = title,
                              color_box = "orange", color_point = "blue", ylim = (-.05,1.05),
                              alpha_point = .5, alpha_box = .3, set_fig_size = False,
                              plot_legend = False)
        
        plt.suptitle("Grid Search Over $\eta$ Values: Model Comparison (K = %d)" % num_topics,
                     fontsize = 30)
        plt.subplots_adjust(top = 0.90)     

        if save_fig:
            if fig_name is None:
                fig_name = "GridEtaScatterbox"
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         fig_override = fig_override,
                         dpi = dpi,
                         bbox_inches = "tight")
        
        
        return [coherences, entropies, kl, phf]







# HELPERS to extract output from WeightedEtaTopicsSearch for GridEtaTopics_scatterbox

# Extraction functions for WeightedEtaTopics search --- need to use this and not
# GridEtaTopics  (note that GridEta_Scatterbox above does use output for
# GridEta) because need some way of picking the best eta for each K
# Thus this visual depends on choice of weights, aggregation method etc. 
# In a way that GridEta_scatterbox, which visualizes each eta model for a given K,
# does not

#Extraction functions for GridEtaTopics




def _extract_num_topics_as_strings(output):
    """Output is dictionary output from WeightedEtaTopicsSearch function"""
    return([str(elem) for elem in list(output.keys())])


def _extract_best_etas_as_strings(output):
    """Output is dictionary output from WeightedEtaTopicsSearch function"""
    etas = [output[elem]["best_eta"] for elem in output.keys()]
    return([str(e) for e in etas])


def _extract_best_coherences(output):
    """Output is dictionary output from WeightedEtaTopicsSearch function,
    returns list of arrays where each array contains the topic coherences
    of the best model for each number of topics used in WeightedEtaTopicsSearch"""
    keys = list(output.keys())
    return([output[k]["best_model_summary"][0] for k in keys])

def _extract_best_entropies(output):
    """Output is output from WeightedEtaTopicsSearch function,
    returns list of arrays where each array contains the topic entropies
    of the best model for each number of topics used in WeightedEtaTopicsSearch"""
    keys = list(output.keys())
    return([output[k]["best_model_summary"][1] for k in keys])

def _extract_best_kl(output):
    """Output is dictionary output from WeightedEtaTopicsSearch function,
    returns list of arrays where each array contains the topic kl divergences from corpus
    of the best model for each number of topics used in WeightedEtaTopicsSearch"""
    keys = list(output.keys())
    return([output[k]["best_model_summary"][2] for k in keys])

def _extract_best_phf(output):
    """Output is dictionary output from WeightedEtaTopicsSearch function,
    returns list of arrays where each array contains the topic phf values
    of the best model for each number of topics used in WeightedEtaTopicsSearch"""
    keys = list(output.keys())
    return([output[k]["best_model_summary"][3] for k in keys])


def _extract_scalertype(output):
    keys = list(output.keys())
    return(output[keys[0]]["scalertype"])

def _extract_aggregation_method(output):
    keys = list(output.keys())
    return(output[keys[0]]["aggregation_method"])

def _extract_weights(output):
    keys = list(output.keys())
    return(output[keys[0]]["weights"])




def GridEtaTopics_scatterbox(output,  metric, topn_coherence = None, topn_phf = None,
                             thresh = None, xtick_rotation = 0,
                             save_fig = False, fig_outpath = None, fig_name = None,
                                      fig_override = False, dpi = 400):
    """
    Plot a boxplot overlaid with points (a "ScatterBox") for each optimized model
    output by WeightedEtaTopics. That is, plot will contain one boxplot per number of topics
    K included in grid search. For each, this will be the boxplot for the model with the
    best eta (in terms of grid search parameters used in call to WeightedEtaTopics)
    among etas considered in GridEtaTopics. The 
    points on the scatterbox represent metrics for the individual topics within each model. 
    There are four metrics available (see below)
 
    Function intention is to provide more insight on the differences between models with
    different numbers of topics
    
 
    Metrics available
    -----------------
        coherence:  'u_mass' coherence of each topic computed using topn_coherence words
        
        phf: stands for "percent high frequency." This is calculated via the
             percent_high_freq() function. It is the percentage of topn_phf
             words in each topic that come from the top (1-thresh)% of words
             by frequency in the overall corpus. Used to examine whether
             topics are mainly relying on high-frequency words
   
        kl: Kullback-Leibler divergence (relative entropy) between the topic 
            and the overall corpus. Larger values mean topic is more distinct
            from corpus-wide probabilities of each word. 0 means they are identical.

        entropy: the entropy (H = -sum_i p_i * ln(p_i)) of each topic. 
            

    Parameters
    ----------
    output : output from WeightedEtaTopics function, without any modifications 
        
    metric : str
        must be one of ['coherence','phf','kl','entropy','all']
        
     **note: topn_coherence, topn_phf, and thresh will have been set in call to
     WeightedEtaTopics that generated output so arguments here are just for 
     plotting purposes!

        topn_coherence : int, optional
            # of words to considered when calculating coherence.
            Must be specified if metric = "all" or "coherence"
            but otherwise can leave default None
         
        topn_phf : int, optional
            # of words to consider when calculating phf 
            must be specified if metric = "all" or "phf"
            but otherwise can leave default None
          
        thresh : int in (0,1), optional
            threshold for phf.
            must be specified if metric = "all" or "phf"
            but otherwise can leave default None

    saving options
    ---------------
    
    save_fig : bool, default False
        if True, saves figure
    
    outpath: str, default None
        if save_fig is True and this is None, saves to working directory
        else, saves to outpath
        
    fig_name : str, default None
         if save_fig is True, saves using this name. If None,
         automatically sets name to GridEtaTopicsScatterbox.
         
         if fig_filename already exists at outpath, adds _1, _2, etc. 
         until finds iflename that does not already exist
         
    fig_override : bool, default is False
        if True, overrides any existing figures with same name
        
    dpi : int, default is 400
        dpi with which to save figures
        
    Returns
    -------
    Returns summary array for whatever metric selected 
    (e.g. for coherence, a list of arrays with each array containing coherences
     for each model in the box plot)
    
    In case of metric = "all" returns list of lists containing these
    [coherences, entropies, kl, phf]
    
    Plots
    -------
    if metric is not "all" it plots a scatterbox of the metric given, comparing
    its spread for each model examined in GridEtaTopic search
    
    if metric is "all" it plots all four metrics in a 4x4 grid

    """
    output_checker(output, func = "WeightedEtaTopicsSearch")
    assert metric in ['coherence','phf','kl','entropy','all'], "metric must be one of ['coherence','phf','kl','entropy','all']"
    
    #don't need the scaler object here. Note that helper functions assume this dictionary
    output = output[0] 
    
    #build labels
    num_topic_vals = _extract_num_topics_as_strings(output)
    eta_vals = _extract_best_etas_as_strings(output)
    labels = [k+"\n("+e+")" for (k,e) in zip(num_topic_vals, eta_vals)]
    
    
    if metric == "coherence":
        assert topn_coherence is not None, "topn_coherence must be specified for coherence"
        coherences = _extract_best_coherences(output)
        
        title = "Spread of Coherence for Topics in each Model"
        title += "\n(using top %d words)" % topn_coherence
        Scatterbox.scatterbox(arr_list = coherences, labels = labels,
                              plot_points = True,
                              xlabel = r"K($\eta$)", ylabel = "Coherence",
                              title = title,
                              color_box = "purple", color_point = "blue",
                              alpha_point = .5, alpha_box = .3, legend_point_label= "Topics",
                              xtick_rotation = xtick_rotation)
        if save_fig:
            if fig_name is None:
                fig_name = "GridEtaTopicsScatterbox"
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         fig_override = fig_override,
                         dpi = dpi,
                         bbox_inches = "tight")
        
        return(coherences)
        
    elif metric == "entropy":
        entropies = _extract_best_entropies(output)
        Scatterbox.scatterbox(arr_list = entropies, labels = labels,
                              plot_points = True,
                              xlabel = r"K($\eta$)", ylabel = "Entropy",
                              title = "Spread of Entropy for Topics in each Model",
                              color_box = "lightgreen", color_point = "blue",
                              alpha_point = .5, alpha_box = .4, legend_point_label= "Topics",
                              xtick_rotation = xtick_rotation)
        if save_fig:
            if fig_name is None:
                fig_name = "GridEtaTopicsScatterbox"
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         fig_override = fig_override,
                         dpi = dpi,
                         bbox_inches = "tight")
        
        return(entropies)
    
    elif metric == "kl":
        kl = _extract_best_kl(output)
        Scatterbox.scatterbox(arr_list = kl, labels = labels,
                              plot_points = True,
                              xlabel = r"K($\eta$)", ylabel = "KL-Divergence from Corpus Distribution",
                              title = "Spread of KL-Divergence for Topics in each Model",
                              color_box = "lightblue", color_point = "blue",
                              alpha_point = .5, alpha_box = .4, legend_point_label= "Topics",
                              xtick_rotation = xtick_rotation)
        if save_fig:
            if fig_name is None:
                fig_name = "GridEtaTopicsScatterbox"
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         fig_override = fig_override,
                         dpi = dpi,
                         bbox_inches = "tight")
       
        return(kl)
    
    elif metric == "phf":
        assert thresh is not None, "thresh must be specified for phf"
        assert topn_phf is not None, "topn_phf must be specified for phf"
        title = "Spread of PHF for Topics in each Model"
        title += "\n(using top %d words and %s threshold)" % (topn_phf, str(thresh))
       
        phf = _extract_best_phf(output)
        Scatterbox.scatterbox(arr_list = phf, labels = labels,
                              plot_points = True,
                              xlabel = r"K($\eta$)", ylabel = "Percentage High Frequency",
                              title = title,
                              color_box = "orange", color_point = "blue", ylim = (-.05,1.05),
                              alpha_point = .5, alpha_box = .3, legend_point_label= "Topics",
                              xtick_rotation = xtick_rotation)
        if save_fig:
            if fig_name is None:
                fig_name = "GridEtaTopicsScatterbox"
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         fig_override = fig_override,
                         dpi = dpi,
                         bbox_inches = "tight")
        
        return(phf)
    
    #----------------------------------all----------------------------------
    else:
        assert thresh is not None, "thresh must be specified for when plotting all metrics"
        assert topn_phf is not None, "topn_phf must be specified when plotting all metrics"
        assert topn_coherence is not None, "topn_coherence must be specified when plotting all metrics"
             
        #get all the values
        coherences = _extract_best_coherences(output)
        entropies = _extract_best_entropies(output)
        kl = _extract_best_kl(output)
        phf = _extract_best_phf(output)
        
        weights = _extract_weights(output)
        grid_info = "scaling: %s,  aggregation: %s, weights: (coherence %s, kl %s)" % (_extract_scalertype(output),
                                                                   _extract_aggregation_method(output),
                                                                   str(weights[0]), str(weights[1]))
        
        #fig1, f1_axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True)
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(20, 20))
        fig.text(0.5, -0.1, 'Number of Topics K (best $\eta$ value for K)' + "\n" + grid_info, 
                 ha='center', fontsize = 26)
        fig.text(-0.05, 0.5, 'Per-Topic Metric', va='center', rotation='vertical', fontsize = 26)
        fig.tight_layout(h_pad=17, w_pad = 10)
    
        plt.subplot(2,2,1)
        title = "Spread of Coherence for Topics in each Model"
        title += "\n(using top %d words)" % topn_coherence
        Scatterbox.scatterbox(arr_list = coherences, labels = labels,
                              plot_points = True,
                              xlabel = r"K($\eta$)", ylabel = "Coherence",
                              title = title,
                              color_box = "purple", color_point = "blue",
                              alpha_point = .5, alpha_box = .3,set_fig_size = False,
                              plot_legend = False, xtick_rotation = xtick_rotation)
        
        plt.subplot(2,2,2)
        Scatterbox.scatterbox(arr_list = entropies, labels = labels,
                              plot_points = True,
                              xlabel = r"K($\eta$)", ylabel = "Entropy",
                              title = "Spread of Entropy for Topics in each Model",
                              color_box = "lightgreen", color_point = "blue",
                              alpha_point = .5, alpha_box = .4,set_fig_size = False,
                              plot_legend = True, legend_point_label= "Topics",
                              xtick_rotation = xtick_rotation)
 
        plt.subplot(2,2,3)
        Scatterbox.scatterbox(arr_list = kl, labels = labels,
                              plot_points = True,
                              xlabel = r"K($\eta$)", ylabel = "KL-Divergence from Corpus Distribution",
                              title = "Spread of KL-Divergence for Topics in each Model",
                              color_box = "lightblue", color_point = "blue",
                              alpha_point = .5, alpha_box = .4,set_fig_size = False,
                              plot_legend = False, xtick_rotation = xtick_rotation)
 

        plt.subplot(2,2,4)
        title = "Spread of PHF for Topics in each Model"
        title += "\n(using top %d words and %s threshold)" % (topn_phf, str(thresh))
       
        Scatterbox.scatterbox(arr_list = phf, labels = labels,
                              plot_points = True,
                              xlabel = r"K($\eta$)", ylabel = "Percentage High Frequency",
                              title = title,
                              color_box = "orange", color_point = "blue", ylim = (-.05,1.05),
                              alpha_point = .5, alpha_box = .3,set_fig_size = False,
                              plot_legend = False, xtick_rotation = xtick_rotation)
        
        plt.suptitle("Grid Search Over $\eta$ and K Values: Model Comparison", fontsize = 30)
        plt.subplots_adjust(top=0.89)
        
        
        if save_fig:
            if fig_name is None:
                fig_name = "GridEtaTopicsScatterbox"
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         fig_override = fig_override,
                         dpi = dpi,
                         bbox_inches = "tight")
        
        
        return [coherences, entropies, kl, phf]



# OUTPUT TYPE CHECK FUNCTION

def output_checker(output, func):
    types = ["GridEta", 
             "GridEtaTopics", 
             "WeightedEtaSearch", 
             "WeightedEtaTopicsSearch"]
    assert func in types, "func must be one of " + str(types)
    
    if func == types[0]:
        assert type(output) == dict, "output is not from GridEta"
        assert list(output.keys()) == ["etas","num_topics","per_topic_summaries","model_list" ], "output is not from GridEta"
    elif func == types[1]:
        assert type(output) == dict, "output is not from GridEtaTopics"
        Kvals = output.keys()
        assert all(type(k) == int for k in Kvals), "output is not from GridEtaTopics"
        for k in Kvals:
            output_checker(output[k], func = "GridEta"),"output is not from GridEtaTopics"
    elif func == types[2]:
        assert type(output) == dict, "output is not from WeightedEtaSearch"
        assert list(output.keys()) == ["num_topics","best_eta","best_combo_val","best_model_summary",
                                      "best_model","etas","combination_vals","scalertype",
                                       "aggregation_method","weights"], "output is not from WeightedEtaSearch"
    elif func == types[3]:
        assert type(output) == tuple, "output is not from WeightedEtaTopicsSearch"
        assert type(output[0]) == dict, "output is not from WeightedEtaTopicsSearch"
        Kvals = output[0].keys()
        assert all(type(k) == int for k in Kvals), "output is not from WeightedEtaTopicsSearch"
        for k in Kvals:
            output_checker(output[0][k], func = "WeightedEtaSearch")
