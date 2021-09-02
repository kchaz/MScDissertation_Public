# -*- coding: utf-8 -*-
"""

This file contains functions to (A) run LDA models with logging and
(B) process log output from gensim LDA logging output 

In particular, it contains a wrapper function that runs a gensim LdaModel() 
call while adding in a logging step and includes functions to process output from that call
    

Author: Kyla Chasalow
Last edited: August 2, 2021


"""
import os
import re
import numpy as np
import logging
from matplotlib import pyplot as plt
import datetime


from gensim.models.ldamodel import LdaModel
from gensim.models.callbacks import PerplexityMetric, ConvergenceMetric, CoherenceMetric

from Helpers import filename_resolver
from Helpers import figure_saver




#### LOG SET-UP

def _log_setup(filename, corpus, outpath = None, topn = 10, 
               callback_list = None,
               resolve_name = True,
               raise_error = False):
    """
    
    set up a log file logging DEBUG level and up. Also set up three callbacks for 
    LDA: coherence, perplexity, and convergence

    Parameters
    ----------
    filename : str
        name of log will be filename + .log
        note that filename.log must not be an existing file
        in directory. If it is, function will raise error
        
    outpath : str, optional, default None
        if outpath is specified, then log of name filename.log will be created
        (if no existing file has same name) in that directory. If
        outpath = None, simply creates it in current working directory  
        
    corpus : gensim corpus
        corpus used to train LDA models
        (vector-count representation)
        
    topn : int, optional
        Number of words to consider when calculating coherence
        in log. The default is 10.
        
    callback_list : list of strings or possibly empty list, optional, default None
        if empty, logging will not track perplexity, coherence, or convergence over epochs
        otherwise, list can contain 1, 2 or all 3 of "coherence","convergence" and "perplexity"
        and will set up those callbacks for the log
        
        Note: if anything other than the three options above is given in list, function will simply
        ignore those

    resolve_name and raise_error:
        provide options for how to deal with the situation that the proposed filename
        already exists. 
        
        1. if resolve_name is True, function will amend filename with _1, _2, _3 etc. until finds 
        one that does not already exist and then use that one. Will do this regardless
        of the value of raise_error
        
        2. if resolve_name is False, there is the possibility of trying to create a log 
        file that already exists. If proposed filename does not already exist, function
        simply creates that log. If it does already exist and raise_error = True, 
        function will raise error. If it does already exist and raise_error = False,
        then existing log file will be overridden
       
    
    ------
    ValueError
        if filename corresponds to an already existing file, function refuses
        to override the old file and intead raises this error

    Returns
    -------
    0. logging object for making further additions to log
    1. file_handle, which allows log to be written to file
    2. list of callbacks metrics set-up and ready to be passed to LdaModel
       call from gensim
       
       
    WARNING
    --------
    For reasons currently not entirely clear to me, using convergence logger 
    leads to MUCH larger file size when saving LDA model. Fine for when 
    running just a few models to test convergence properties but not good 
    when running grid search. There, it seems that the gensim model.save() 
    is saving some aspect of these callback calculations  for all the models it fits. 
    I infer this because the file size for the saved model from grid search
    (see LdaGridSearch.py) gets larger the more eta values searched over (the more models fit). 
    
    My tentative explanation is that because in grid search, I set-up a single log for
    searching over all the eta values (for a given num_topics), a single object is created
    for each callback metric (e.g. a single convergence logger). This single object
    accumulates information at each epoch of training for each model and all of this somehow
    gets saved when any one of the models is saved
    
    Perplexity and Coherence also increase file size but to much lesser degree. 
    E.g. on Toy example, model with no loggers was 4 KB, with just perplexity 5 KB, 
    with just coherence 5 KB, and with just convergence, 47 KB. With all three, 54 KB.

    """
    assert type(callback_list) == list or callback_list is None, "callback_list must be a list or None"
    assert type(filename) == str, "filename must be a string"
    assert type(topn) == int, "topn must be an integer"
 
   
    #if log file does not already exist, then filename resolver will just return filename
    #if does already exist, will amend it
    if resolve_name:
         filename = filename_resolver(filename = filename, 
                                         extension = "log", 
                                         outpath = outpath)   
    
    #if not resolving name and not raising error, then will either create a
    #totally new file (no need to remove anything) or will remove existing
    #file and create new one with same name
    elif not raise_error: 
        #delete existing log file to override it...only if that file already exists
        try:
            if outpath is None:
                os.remove(filename + ".log")
            else:
                os.remove(os.path.join(outpath,filename + ".log"))
        except:
            pass #do nothing

    #if not resolving name but have specified that want to raise error, then
    #check if file already exists. If it does, raise error, if it doesn't move on
    elif raise_error:
        #join outpath and filename if outpath specified
        if outpath is not None:
            filepath = os.path.join(outpath, filename + ".log")
        else:
            filepath = filename + ".log"
       
        #check if file already exists and raise error if it doesn't
        try:
            open(filepath, "r")
            file_problem = True
        except Exception:
            file_problem = False
            
        if file_problem:
            message = "You requested that I create a log called '" + filename + ".log'"
            if outpath is not None:
                message = message + "\n at:" + filepath
            message = message + "\n That file already exists. Please choose a different filename."
            raise ValueError(message)
    
    #if get to this point, then we have either original or resolved filename to work with 
    if outpath is not None:
            filepath = os.path.join(outpath, filename + ".log")
    else:
            filepath = filename + ".log"
    
    #CREATE LOG FILE
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    
    file_handle = logging.FileHandler(filename= filepath)
    file_handle.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
                        fmt='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                        )
    file_handle.setFormatter(formatter)
    log.addHandler(file_handle)
        
    # Set up the callbacks for logging
    if callback_list is not None:
        callbacks = []
        if "perplexity" in callback_list:
            perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
            callbacks.append(perplexity_logger)
        if "convergence" in callback_list:
            convergence_logger = ConvergenceMetric(logger='shell', distance = "kullback_leibler")
            callbacks.append(convergence_logger)
        if "coherence" in callback_list:
            coherence_umass_logger = CoherenceMetric(corpus=corpus, logger='shell', coherence = 'u_mass',                                                 topn = topn)
            callbacks.append(coherence_umass_logger)
        #if argument that didn't include any of above given, will just rever to None
        if callbacks == []:
            callbacks = None  
    else:
        callbacks = None
    
    return(log, file_handle, callbacks)
   




#### FUNCTION FOR CREATING LOG, RUNNING LDA, CLOSING LOG

def LdaModelLogged(log_filename, log_outpath = None, resolve_name = True, raise_error = False,  #log options
                   save_model = False, model_fname = "", model_outpath = None, #model saving options
                   callback_list = ["coherence","perplexity", "convergence"],
                   num_topics=100, corpus=None, id2word=None,
                   distributed=False, chunksize=2000, passes=1, update_every=1,
                   alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10,
                   iterations=50, gamma_threshold=0.001, minimum_probability=0.01,
                   random_state=None, ns_conf=None, minimum_phi_value=0.01,
                   per_word_topics=False, dtype=np.float32, topn_coherence = 10):
    """
    
    Wrapper function for gensim LdaModel that:
            0. creates log file
            1. initializes gensim callbacks 
            2. runs LDA model while storing logging output in file (DEBUG level and up)
            3. closes log file
            4. returns LDA model
    
    Logging has option to use some or all of following three callbacks
        perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
        convergence_logger = ConvergenceMetric(logger='shell')
        coherence_umass_logger = CoherenceMetric(corpus=corpus, logger='shell', coherence = 'u_mass')
    
    Does not return log file but creates it in working directory, writes to it, and closes it
   
    
    Parameters
    ----------
    
    ------logging options----------
    
    log_filename : str
        name of logging file will be log_finename.log
        note: file name must not already exist
 
    log_outpath : str, optional, default None
        if outpath is specified, then log of name filename.log will be created
        (if no existing file has same name) in that directory. If
        outpath = None, simply creates it in current working directory  

    resolve_name and raise_error
        for logging, provides two options for how to deal with the situation that the proposed filename
        already exists. In neither case will it override the file.
        
        if resolve_name = True, will amend filename with _1, _1_2, _1_2_3 etc. until finds 
        one that does not already exist
        
        if resolve_name = False and raise_error = True, then will raise an error if the filename
        already exists
        
    callback_list : list of strings or None, optional, default all three options
    
        if None, logging will not track perplexity, coherence, or convergence over epochs
        otherwise, list can contain 1, 2 or all 3 of "coherence","convergence" and "perplexity"
        and will set up those callbacks for the log
        
        Note: if anything other than the three options above is given in list, function will simply
        ignore those


    topn_coherence : int, optional
        Number of words to consider when calculating coherence
        in log. The default is 10.
        
    ----------model saving options---------------------
    
    save_model : bool, optional
        optionally, save model. Default is False
        
    model_fname : str, optional
        if save_model = True, will require that save_name is not none
        will save model using save_name and appending _model to the end of it
        Default is "" and in that case, name will be just _model
        
    model_outpath : str, optional
        optionally specify a file path where model should be saved.
        If None, saves to working directory.
     
    note: will never override an existing model - uses filename resolver if
    proposed name already exists


    rest of parameters are copied over from LdaModel with same defaults as in
    in gensim with the EXCEPTION of 'callbacks' which is specified below
    https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/ldamodel.py
 
    
    Raises
    ------
    ValueError
        if filename corresponds to an already existing file, function refuses
        to override the old file and intead raises this error


    Returns
    -------
    Gensim LdaModel object with given parameters
    
    Returns None and raises error if filename already exists in working directory 
    - will not write log to existing log

    """

    #set-up logging, including check if file already exists
    log, file_handle, callbacks = _log_setup(filename = log_filename,
                                             corpus = corpus,
                                             outpath = log_outpath,
                                             topn = topn_coherence,
                                             callback_list = callback_list,
                                             raise_error= raise_error,
                                             resolve_name = resolve_name)
    
    model = LdaModel(
        corpus = corpus,
        id2word = id2word,
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
    
    log.info('-------Closing Log-------')
    log.removeHandler(file_handle)
    del log, file_handle
    
    
    if save_model:
        model_fname = filename_resolver(model_fname + "_model", extension = None,
                                        outpath = model_outpath)
        # add outpath if needed
        if model_outpath is not None:
            model_fname = os.path.join(model_outpath, model_fname)
        model.save(model_fname)
    
    return(model)

 
    
    



#### FUNCTIONS FOR LOADING LOGS AND EXTRACTING VARIOUS ASPECTS OF TOPIC MODEL


# TO DO: ADD CHECKS FOR WHETHER METRIC ACTUALLY IN LOG
# E.g. if fun function with only some callbacks, should
# intelligently tell you if cannot find that output in the log


def load_log(filename, path = None):
    """Function to load a log file and return a 
    list version with one line of log per entry
    
    filename should include .log
    
    optionally, specify path where log is located
    """
    if path is not None:
        filename = os.path.join(path, filename)
    file = open(filename, "r")
    log = file.read()
    return(log.split("\n"))


def _extract_K(log_list):
    """
     log_list : list
        list representing a single log. Contains one line of a log per entry
        e.g. as output by load_log()
        
        ***note: assumes log contains only output from run of ONE model
        
    Returns
    ---------
    Number of Topics, as extracted from log
    """
    K = [int(e) for entry in log_list for e in re.findall(r'(\d+) topics', entry)][0]
    return(K)


def _extract_group_size(log_list):
    """
     log_list : list
        list representing a single log. Contains one line of a log per entry
        e.g. as output by load_log()
        
        ***note: assumes log contains only metrics for
        run of ONE model
        
    Returns
    ---------
    Number of groups of documents processed in training 
    This depends on the chunksize and size of the corpus. For example,
    if corpus is 2400 documents, chunksize = 1000 results in 3 groups (1000, 1000, 400)
    """
    chunks = [int(e) for entry in log_list for e in re.findall(r'chunk of (\d+) documents', entry)]
    group_size = np.where(chunks < np.max(chunks))[0][0] + 1 #only last one will be less than max, +1 b/c 0 counting
    return(group_size)

 
def _extract_mean_coherence(log_list, epoch = None):    
    """
    Extract the mean coherence at specified epoch or if no epoch specified,
    from the last one in the log
    """
    coherence_entries = [e for e in log_list if re.search(r'Epoch \d+: Coherence estimate:', e)]
    coherences = [float(num) for entry in coherence_entries for num in re.findall(r'Coherence estimate: (.?\d+\.\d+)', entry)]
    if epoch is None:
        epoch = len(coherences) - 1 #get last 
    return(coherences[epoch])

   




#### FUNCTIONS FOR VISUALIZING LOG PROGRESS

    
def _progress_plotter(epochs, values, ylabel, color = "blue", title = ""):
    """
    Parameters
    ----------
    epochs : np.array or list of epochs (x values)
    values : np.array or list of corresponding metric (y values)
    ylabel : str containing name of metric to plot on y axis
    color : str, optional
        scatterplot color. The default is "blue".
    
    title : optionally, specify a title

    Returns
    -------
    None.

    """
    plt.scatter(epochs, values, color = color)
    plt.plot(epochs, values, color = color)
    plt.title(title, pad = 15, fontsize = 22)
    plt.ylabel(ylabel, fontsize = 16)
    plt.xlabel("Epochs", fontsize = 16)
    plt.yticks(fontsize = 13)
    plt.xticks(fontsize = 13)

    




def extract_metric(log_list, metric, plot = False, title = ""):
    """
    
    Parameters
    ----------
    log_list : list
        list representing a single log. Contains one line of a log per entry
        e.g. as output by load_log()
        
        ***note: assumes log contains only metrics for
        run of ONE model
    
    metric : str
        must be one of ['perplexity','Perplexity',
                        'coherence','Coherence',
                        'convergence','Convergence',
                        'all']
        capitalization only affects what is plotted on y axis of plots
        
        
    plot : bool
        optionally plot metric(s) over epochs
        default False
        if metric = "all", plots 3 separate plots, one for each metric
        
    title : optional title for plots (will not be included for "all" option)
        
    Returns
    -------
        0. np.array of epoch #'s
        1. if one metric is given, np.array of metric, in order of epoch. If metric == 'all', 
           this is a tuple containing an array of three such arrays in order of (perplexity, coherence, convergence)

    """
    
    assert metric.lower() in ['perplexity','coherence','convergence', 'all'], "Invalid metric argument"

    #obtain metric values and epochs from log using regex; the first part of (.?\d+\.\d+) is to allow for negatives
    perplexity_entries = [e for e in log_list if re.search(r'Epoch \d+: Perplexity estimate:', e)]   
    perplexities = [float(num) for entry in perplexity_entries for num in re.findall(r'Perplexity estimate: (.?\d+\.\d+)', entry)]
   
    coherence_entries = [e for e in log_list if re.search(r'Epoch \d+: Coherence estimate:', e)]
    coherences = [float(num) for entry in coherence_entries for num in re.findall(r'Coherence estimate: (.?\d+\.\d+)', entry)]
    
    convergence_entries = [e for e in log_list if re.search(r'Epoch \d+: Convergence estimate:', e)]
    convergences= [float(num) for entry in convergence_entries for num in re.findall(r'Convergence estimate: (.?\d+\.\d+)', entry)]
    
    #assuming all have same # of epochs...should be gauranteed by how LDA model works
    epochs = np.unique([int(ep) for entry in log_list for ep in re.findall(r'Epoch (\d+):', entry)])

    if plot:
        plt.rcParams.update({'font.family':'serif'})
        
        if metric.lower() == "perplexity":
             plt.figure(figsize=(12,5))
             _progress_plotter(epochs, perplexities, metric, color = "green", title = title)
                          
        elif metric.lower() == "coherence":
             plt.figure(figsize=(12,5))
             _progress_plotter(epochs, coherences, "Mean Topic Coherence", color = "purple", title = title)
             
        elif metric.lower() == "convergence":
             plt.figure(figsize=(12,5))
             _progress_plotter(epochs, convergences, "Convergence", color = "blue", title = title)
             
        elif metric == "all":
            r = 3; c = 1
            plt.figure(figsize=(10,12))
            plt.subplot(r,c,1)
            _progress_plotter(epochs, perplexities, "Perplexity", color = "green")

            plt.figure(figsize=(10,12))
            plt.subplot(r,c,2)
            _progress_plotter(epochs, coherences, "Mean Coherence", color = "purple")

            plt.figure(figsize=(10,12))
            plt.subplot(r,c,3)
            _progress_plotter(epochs, convergences, "Convergence", color = "blue")
            
              
    if metric.lower() == "perplexity":
        return(epochs, np.array(perplexities))
    elif metric.lower() =="coherence":
        return(epochs, np.array(coherences))
    elif metric.lower() =="convergence":
        return(epochs, np.array(convergences))
    elif metric == "all":
        return(epochs, np.array([perplexities, coherences, convergences]))
   


   
    
    
def doc_convergence_report(log_list, group_size = 1, plot = False):
    """
    extract information about the percentage of documents that have
    converged in variational inference algorithm for each chunk of documents
    within each epoch (pass) of the algorithm. Outputs a simple ordered array
    of percentages which can be used to evaluate whether LDA iteration and epochs
    arguments have been set high enough. The percentage of converged documents
    should generally increase with epochs.
    
    
    
    Parameters
    ----------
    log_list : list
        list containing one line of a log per entry
        e.g. as output by load_log()
        
        ***note: assumes log contains only metrics for run of ONE model
    
    group_size : int
        optionally, specify how entries should be grouped. E.g. if know that 
        there were 10 chunks in each pass, might specify group_size = 10. Output
        percentages will then be one per pass rather than one per chunk. 
        default is 1, for which it calculates percentage for each entry individually
        
        ***note: total number of document convergence log entries 
        must be evenly divisible by group_size
        
    plot : bool
        optionally plot convergence over entries of report. Note that this is 
        more meaningful if have specified appropriate group_size so that each
        entry corresponds to a pass over whole corpus 

    Returns
    -------
    np.array of percentages
    
    
    Example of type of log entries processed:
        
     2021-06-30 12:19:10,575 : DEBUG : 6/10 documents converged within 100 iterations,
     2021-06-30 12:19:10,592 : DEBUG : 1/4 documents converged within 100 iterations,
     2021-06-30 12:19:10,621 : DEBUG : 10/10 documents converged within 100 iterations,
     2021-06-30 12:19:10,635 : DEBUG : 4/4 documents converged within 100 iterations,
     
     output here would be np.array containing [0.6,0.25,1.0,1.0] for group_size =1
     and [0.5,1.0] for group_size = 2. There are 14 documents total and the above shows 
     two passes through them.

    """

    #get log entries
    doc_report = [e for e in log_list if re.search(r'DEBUG: .+ documents converged within', e)]
    #get entries of form (4/4)
    fracs = [frac for entry in doc_report for frac in re.findall(r'(\d+/\d+)', entry)]
    #separate into pairs of integers
    pieces = [list(map(int,elem.split('/'))) for elem in fracs]
    
    #check groupsize is valid
    assert len(pieces) % group_size == 0, "Number of entries must be evenly divisible by group_size"
    
    #group them into group_size groups and sum numerator and denominator over each
    num_groups = len(pieces) // group_size
    out = []
    for i in range(num_groups):
        out.append(pieces[group_size * i].copy())
        for j in range(1,group_size):
            out[i] += pieces[group_size*i+j].copy()
    
    #percentages
    perc_docs_converged = [elem[0]/elem[1] for elem in out]
    
    
    if plot:
        plt.rcParams.update({'font.family':'serif'})
        plt.figure(figsize=(12,5))
        plt.scatter(range(len(perc_docs_converged)), perc_docs_converged)
        plt.plot(range(len(perc_docs_converged)), perc_docs_converged)
        plt.title("Document-Level Optimization", 
                  pad = 15, fontsize = 20)
        plt.ylabel("Percent of Docs Converged", fontsize = 16)
        plt.xlabel("Index", fontsize = 16)
        plt.yticks(fontsize = 13)
        plt.xticks(fontsize = 13)
    
    
        
    return(perc_docs_converged)

   

    
### function to just plot document convergences
def doc_convergence_plot(filenames_lst, path = None, group_size = "auto", zoom = 0, labels = None,
                               set_fig_size = True, plot_title = True, figsize = (15,6),
                               xlabel = "Epoch"):
    """
    Plotting function
    
    Compare multiple document-level convergence trajetories at once for models trained
    with same group_size (a function of chunksize in gensim - see doc_convergence_report
    documentation) and same number of epochs but potentially different numbers of iterations
    or other parameters
    
    
    Optionally, discard up to the zoom^th value of percent_docs_converged from doc_convergence_report
    in order to zoom in on fluctuations in # of documents converged after the
    usually-very-poor first epoch result
    
    
    Requirements
    --------------
    1. Filenames must correspond to models with logs saved either in given path
        or, if path is not specified, in current working directory
    2. models must all have same number of epochs
    3. models must all have been trained with same chunksize argument 

    
    Parameters
    ----------
    filenames_lst : list of strings
        list of filenames for .log files from which to extract
        info about metric for each model. Note, filename should not
        include ".log"
        
    path : optionally specify path where files are located
        
    group_size : int or "auto", default is auto
        
        if auto, will automatically detect group_size from the first log in filenames_lst
        so that points plotted are the percentage of documents converged from the entire corpus in each epoch
        (divided into however many chunks in training)
        
        optionally, can specify group_size specify how entries should be grouped.  
        
        ***note: total number of document convergence log entries 
        must be evenly divisible by group_size
        

    zoom : int >= 0
         remove the first zoom values so as to zoom in on graph by removing
         earlier, less converged values
         
    labels : str
        optionally specify labels for each file to be used in legend
        if unspecified, filenames are used
        

    set_fig_Size : bool, optional, default is True
        in general, function will set an appropriate figure size 
        for the figure but if plotting these box plots as part of a grid
        need to turn this off so that figsize setting doesn't conflict
        with grid settings. 
        
    figsize : tuple of two values, default is (12,7)
        controls dimensions of resulting plot, as long as set_fig_size = True
        
    plot_title : bool, optional, default if True
        if False, does not plot title
        This option exists to make it easier to add this plot to a grid
        
    xlabel : str, 
        label for x axis of plot, default is "Epoch"

    Returns
    -------
    list of lists of output from doc_convergence_report for each file in filenames_lst

    """
    assert type(zoom) == int and zoom >= 0, "zoom must be a non-negative integer"
    if type(group_size) == int:
        assert group_size > 0, "group_size must be a positive integer"
    elif type(group_size) == str:
        assert group_size == "auto",  "group_size must either be an integer or be 'auto'"
    
    if labels == None:
        labels = filenames_lst
        
  
    log_lists = [load_log(f + ".log", path = path) for f in filenames_lst]
   
    if group_size == "auto":
         group_size = _extract_group_size(log_lists[0])
    
   
    percents_list = [doc_convergence_report(lst,
                                            group_size = group_size,
                                            plot = False) for lst in log_lists]
 
    #check lengths
    len_list = [len(elem) for elem in percents_list]
    assert all(elem == len_list[0] for elem in len_list), "models must have same number of epochs" 
    num_epochs = len(percents_list[0]) 
    #depending on if group_size set correctly, this might not actually be # of epochs
        
    plt.rcParams.update({'font.family':'serif'})

    if set_fig_size:
        plt.figure(figsize = (15,6))
    
    for i in range(len(percents_list)):
        plt.scatter(range(zoom, num_epochs), percents_list[i][zoom:], label = labels[i])
        plt.plot(range(zoom, num_epochs),  percents_list[i][zoom:])
    
    plt.xlabel(xlabel,fontsize = 20)
    plt.ylabel("Percent of Docs Converged", fontsize = 20)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
        
    if zoom > 0:
        addition = " (zoomed in)"
    else:
        addition = ""
        
    if plot_title:
        plt.title("Document-Level Optimization" + addition, pad = 15, fontsize = 22)
    plt.legend(fontsize = 18)
    
    
    return(percents_list)






#Function to plot perplexity, coherence, convergence, or all 3 + document convergence as well 
def metric_comparison_plot(filenames_lst, metric, path = None,  labels = None, legend_title = "",
                            save_fig = False, fig_outpath = None, 
                            fig_name = "convergence_plot", dpi = 200,
                            fig_override = False):
    """
    
    plots metric over epochs for all models in filenames_list on same plot.
   
    if "all", plots perplexity, coherence, convergence 
    AND ALSO % documents converged (see doc_convergence_plot()) on a 4x4 grid
    
    Requirements
    --------------
    1. Filenames must correspond to models with logs saved in current working directory
    2. models must all have same number of epochs
    3. models must all have been trained with same chunksize argument 
    
    
    Parameters
    ----------
    filenames_lst : list of strings
        list of filenames for .log files from which to extract
        info about metric for each model. Note, filename should not
        include ".log"
        
    metric : str
        must be one of ['perplexity','Perplexity',
                        'coherence','Coherence',
                        'convergence','Convergence',
                        'all']
        capitalization only affects what is plotted on y axis of plots

    path : str
        optionally, specify path where files in filename_lst are located

    labels : str
        optionally specify labels for each file to be used in legend
        if unspecified, filenames are used
        
    legend_title : str, optional
        title for legend containing labels

    Returns
    -------
    list of np.arrays containing metrics over epochs for each file in filenames_list

    """
    assert metric.lower() in ['perplexity','coherence','convergence', 'all'], "invalid metric given"
    
    num_models = len(filenames_lst)
    
    
    if labels == None:
        labels = filenames_lst
 
    #load the logs in list format
    load_log(filenames_lst[0] + ".log", path = path)
    log_lists = [load_log(f + ".log", path = path) for f in filenames_lst]


    if metric != "all":    
        #obtain values of metric from log for each model log in log_list
        info_list = [extract_metric(l, metric, plot = False)[1] for l in log_lists]
        
        
        #check lengths
        len_list = [len(elem) for elem in info_list]
        assert all(elem == len_list[0] for elem in len_list), "models must all have same number of epochs"
        num_epochs = len_list[0]
    
        #plot
        plt.rcParams.update({'font.family':'serif'})
        plt.figure(figsize = (15,6))
        for i in range(num_models):
            plt.scatter(range(num_epochs), info_list[i], label = labels[i])
            plt.plot(range(num_epochs), info_list[i])
        
        plt.xlabel("Epoch",fontsize = 18)
        plt.ylabel(metric, fontsize = 18)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        if metric.lower() == "coherence":
            plt.ylabel("Mean Coherence", fontsize = 18)
        if metric.lower() == "convergence":
            plt.ylabel("Topic Convergence", fontsize = 18)
        plt.title(metric.capitalize() + " over Epochs", pad = 15, fontsize = 22)
        plt.legend(fontsize = 15, title = legend_title)
        
            #optionally save figure
        if save_fig:
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                         bbox_inches = "tight")
    
        
    elif metric == "all":
        
        
        info_list = [extract_metric(l, metric, plot = False)[1] for l in log_lists]
            
        #check lengths
        len_list = [len(elem[0]) for elem in info_list]
        assert all(elem == len_list[0] for elem in len_list), "models must all have same number of epochs"
        num_epochs = len_list[0]  
        epochs = range(num_epochs)
        
        #extract each kind of info
        perplexities = [elem[0] for elem in info_list]
        coherences = [elem[1] for elem in info_list]
        convergences = [elem[2] for elem in info_list]
          
        #global plotting parameter
        plt.rcParams.update({'font.family':'serif'})
      
        #set up 4 x 4 grid
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(20, 12))
        fig.text(0.5, 0.04, 'Epochs', ha='center', fontsize = 26)
        fig.text(0.04, 0.5, 'Metric', va='center', rotation='vertical', fontsize = 28)
        
        #perplexities
        plt.subplot(2,2,1)
        for i in range(num_models):
            plt.scatter(epochs, perplexities[i], label = labels[i])
            plt.plot(epochs, perplexities[i])
        
        plt.ylabel("Training Perplexity", fontsize = 20)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        
        #coherences
        plt.subplot(2,2,2)  
        for i in range(num_models):
            plt.scatter(epochs, coherences[i], label = labels[i])
            plt.plot(epochs, coherences[i])
        
        plt.ylabel("Mean Coherence", fontsize = 20)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        
        #convergence
        plt.subplot(2,2,3) 
        for i in range(num_models):
            plt.scatter(epochs, convergences[i], label = labels[i])
            plt.plot(epochs, convergences[i])
        plt.ylabel("Topic Convergence", fontsize = 20)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        
        #documents - only this one gets the legend
        plt.subplot(2,2,4)
        g = _extract_group_size(log_lists[0])
        doc_convergence_plot(filenames_lst, path = path, group_size = g, zoom = 0, labels = labels,          # TO DO: GET GROUP SIZE FUNCTION
                               set_fig_size = False, plot_title = False, xlabel = "")
        
        plt.suptitle("LDA Training Plots", fontsize = 30)
        
        if save_fig:
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                         bbox_inches = "tight")
    
            
    
    return(info_list)

   
    


def plot_mean_coherence_comparison(log_lists, epoch = None, 
                                   connect = True, color = "purple", 
                                  figsize = (10, 6), save_fig = False, fig_outpath = None, 
                                  fig_name = "mean_coherence_plot", dpi = 200,
                                  fig_override = False):
    """
    Simple scatter(-line) plot plot comparing K values of each log in log_lists to mean coherence
    of model represented by log at specified epoch. If epoch not specified,
    will use last epoch in each log. In principle, logs should then be of
    same length but function will work even if not.

    Parameters
    ----------
    log_lists : a list of "log lists" (logs already converted to one-line-per-entry
                                       list form) as output by load_log()
    epoch : int, optional
        epoch from which to extract mean coherences from log. The default is None.
    connect : bool, optional
        if True, connects points via a line 
    color : str, optional
        color of plot. The default is "purple".
    figsize : tuple, optional
        optionally adjust figure size. The default is (10, 6).

    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

    Returns
    -------
    None.

    """
    #extracting values
    coherences = [_extract_mean_coherence(log_list, epoch = epoch) for log_list in log_lists]
    Kvals = [_extract_K(log_list) for log_list in log_lists]
    # Creating the plot
    plt.rcParams.update({'font.family':'serif'})
    plt.figure(figsize = figsize)
    plt.scatter(Kvals, coherences, color = color, s = 90)
    if connect:
        plt.plot(Kvals, coherences, color = color, alpha = 0.5)
    plt.xticks(ticks = Kvals, labels = Kvals, fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.ylabel("Mean Coherence", fontsize = 18)
    plt.xlabel("K", fontsize = 18)
    plt.title("Mean Coherence vs Number of Topics", fontsize = 20, pad = 15)
    plt.xlim(0,) #start x axis at 0 for context
    
    #optionally save figure
    if save_fig:
        figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     dpi = dpi,
                     fig_override = fig_override,
                     bbox_inches = "tight")







##### Computing Time ANALYSIS


def get_log_duration(log):
    """

    extract how much time is covered by a log
    
    NOTE: assumes log contains datetime formatting of form %Y-%m-%d %H:%M:%S

    Parameters
    ----------
    log : str or list
        if string of the form "name.log", will try to read in this file and turn it into a list
        with one entry per line 
        
        if string without .log extension, will assume this is a raw log file, read in as a string
        but not yet split into a list
        
        if type is list, assumes this is a list of strings with each entry representing line of log
        
    Returns
    -------
    Amount of time in seconds covered by log

    """
    
    #if have been given name of a log file
    if type(log) == str and log.find(".log") != -1:
        i = log.find(".log")
        assert log[i:] == ".log", "argument contains '.log' but this is not the end of the string"
        file = open(log, "r")
        log = file.read()
        log_list = log.split("\n")
    
    #if have been given read-in log file in string form
    elif type(log) == str:
        log_list = log.split("\n")
    
    #if have been given read-in log in list form (each line an entry)
    elif type(log) == list:
        log_list = log

    #extract just the datetimes and get rid of any emptylists
    datetimes = [re.findall(r"(.+) INFO",e) for e in log_list]
    datetimes = [elem[0] for elem in datetimes if elem != []]
    
    starttime = datetime.datetime.strptime(datetimes[0], '%Y-%m-%d %H:%M:%S')
    endtime = datetime.datetime.strptime(datetimes[len(datetimes)-1], '%Y-%m-%d %H:%M:%S')
    dif = endtime - starttime
    return(dif.total_seconds())


def get_all_log_durations(list_of_logs):
    """
    
    applies get_log_duration to every log in a list of logs

    Parameters
    ----------
    list_of_logs : list of strings or lists
        list of logs
        see get_log_duration() documentation for forms entries can take

    Returns
    -------
    durations

    """
    times = [get_log_duration(elem) for elem in list_of_logs]
    return(times)



def comptime_comparison_plot(xvals, times, xlabel, title, ylim = None,
                            color = "blue", figsize = (8,6), set_figsize = True,
                            xtick_rotation = 0, plot_ylabel = True,
                            set_xticks_to_xvals = False):
    """
    plot simple scatterplot of xvalues representing models
    (could be # of epochs, K values, eta values etc.)
    versus runtimes for each model

    Parameters
    ----------
    xvals : list or array of x values for plot
    times : list or array of times, of same length as xvals
    xlabel : str, x axis label
    title : str, title
    ylim : optionally, specify y limits. If none, sets them to 0 and 1.1 * max(times)
    color : str, optional, default blue
    figsize : tuple, optional, default (8,6)
    set_figsize : bool, optional, default True. If False, does not set figsize
    xtick_rotation : int, optional, default 0
    plot_ylabel: bool, optional, turn off y label, e.g. for gridding
    set_xticks_to_xvals : bool, optional, default False. If True, sets x ticks to x vals 

    Returns
    -------
    None.

    """
    if set_figsize:
        plt.figure(figsize = figsize)
    plt.scatter(xvals, times, s = 80, color = color)
    plt.plot(xvals, times, alpha = 0.3)
    plt.xlabel(xlabel, fontsize = 18)
    if plot_ylabel:
        plt.ylabel("Time (seconds)", fontsize = 25)
    plt.title(title, pad = 20, fontsize = 20)
    plt.xticks(fontsize = 15, rotation = xtick_rotation)
    plt.yticks(fontsize = 15)
    if set_xticks_to_xvals:
        plt.xticks(ticks = xvals, labels = xvals)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        plt.ylim(0,np.max(times)*1.1)
















    