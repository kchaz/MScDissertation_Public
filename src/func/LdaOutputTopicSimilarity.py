# -*- coding: utf-8 -*-
"""

This file contains functions for processing the output of gensim LdaModels

Specifically, it contains functions for calculating and visualizing topic similarity
and topic correlations


Author: Kyla Chasalow
Last edited: July 31, 2021


"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Scatterbox

from Helpers import figure_saver
from Helpers import filename_resolver


############## Topic Similarities / Differences   

def _topic_dif_vec(model, distance):
    """
    
    Parameters
    ----------
    model : gensim LDA model object
    
    distance : one of "kullback_leibler" or "jensen_shannon"
    
    num_words : number of words from topic to consider in making comparison

    Returns
    -------
    flattened vector of unormalized KL divergences or JS divergences between all pairs of topics

    """
    sim_mat,_ = model.diff(model, distance = distance,  
                           annotation = False, 
                           normed = False)
    lower_tri = np.tril(sim_mat).flatten()
    sim = lower_tri[lower_tri != 0]
    return(sim)



def get_all_topic_difs(model_list, distance,
                       save_array = False, outpath = None,
                       filename = "topic_difference_array",
                       override = False):
    """ Apply _topic_dif_vec to every model in model list"""
    assert distance in ["kullback_leibler", "jensen_shannon"]
    model_difs = [_topic_dif_vec(m, distance = distance) for m in model_list]
    
    if save_array:
        if not override:
            filename = filename_resolver(filename = filename,
                                       extension = "pickle",
                                       outpath = outpath)
        if outpath is not None:
            filepath = os.path.join(outpath, filename + ".pickle")
        else:
            filepath = filename + ".pickle"
            
        with open(filepath, "wb") as handle:   
            pickle.dump(model_difs, handle) 
    
    return(model_difs)



def visualize_topic_difs(model_dif_list, labels, distance, xlabel = "",   
                         color_box = "red", color_point = "blue",
                         alpha_box = .45, alpha_point = .3, ylim = None,
                         save_fig = False, fig_outpath = None, 
                         fig_name = "topic_dif_plot", dpi = 200,
                         fig_override = False):
    """
    wrapper applying scatterbox() function (see documentation for parameters there)
    to topic similarities 
    
    model_dif_list is as output by get_all_topic_difs
    
    Returns
    -------
    None

    """
    assert distance in ["kullback_leibler", "jensen_shannon"]
    assert len(model_dif_list) == len(labels), "labels must be same length as model list"
    
    if distance == "kullback_leibler":
        ylab = "KL Divergences Between Topics"
    else:
        ylab = "JS Divergences Between Topics"
    
    
    Scatterbox.scatterbox(arr_list = model_dif_list, labels = labels, plot_points = True,
                              xlabel = xlabel, ylabel = ylab,
                              title = "Comparing Spread of Differences Between Topics for each Model",
                              color_box = color_box, color_point = color_point,
                              alpha_point = alpha_point, alpha_box = alpha_box,
                              ylim = ylim, legend_point_label = "Pairs of Topics")
    
    #optionally save figure
    if save_fig:
        figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     dpi = dpi,
                     fig_override = fig_override)
    

      
    
    

# following function inspired by
#  https://radimrehurek.com/gensim/auto_examples/howtos/run_compare_lda.html#sphx-glr-auto-examples-howtos-run-compare-lda-py
    
def _plot_heatmap(dif_mat, title="", ylabel = "", xlabel = "",
                 figsize = (16,12), cmap = "RdBu_r"):
    """Helper to Plot heatmap of difference between model topics"""
    fig, ax = plt.subplots(figsize=figsize)
    #origin = lower flips matrix so even though topic 0 vs topic 0 is in upper left corner of matrix
    #it ends up in bottom left corner of heatmap
    data = ax.imshow(dif_mat, cmap=cmap, origin='lower') 
    plt.title(title, pad = 20, fontsize = 25)
    plt.xlabel(xlabel, fontsize = 22)
    plt.ylabel(ylabel, fontsize = 22)
    plt.colorbar(data)
    
    
    
    
def plot_topic_heatmap(pmodel, qmodel,  distance, normed = False, cmap = "RdBu_r",
               figsize = (16,12), save_fig = False, fig_outpath = None, 
               fig_name = "topic_heatmap", dpi = 200,
               fig_override = False):
    """
    Plot heatmap of difference between model topics
    in terms of either KL divergence or
    Jensen-Shannon Divergence
    
    IF jensen_shannon:
        pmodel is y axis model
        qmodel is x axis model
        but for the rest, it makes no difference which model is which
    
    If kullback_leibler
        p and q refer to which model's topics are treated as
        the p distribution in KL divergence which as the q where
    
        KL = p(x) log (p(x)/q(x))
    
        the p model will be put on the y axis and the q model
        on the x axis.

    Optionally, set normed = True to normalize to be between [0,1]
    over the matrix (this does mean that if have multiple heatmaps
    won't be able to tell if one has systematically higher/lower values)

    Parameters
    ----------
    pmodel : Gensim LDA model

    qmodel : Gensim LDA model
    
     cmap : str, optional
        specify color map for heatmap. The default is 

    figsize : bool, optional
        set figure size. The default is (16,12).
        
    distance : str, one of "kullback_leibler" and "jensen_shannon"
    
    normed : if True, normalizes values over whole matrix to be in [0,1] - erases any
        systematic differences between different matrices/heatmaps


    *note: if pmodel and qmodel are the same model, plots only upper triangle and diagonal is left
    blank to avoid the perfect similarity making the rest of the gradiations in the graph
    indistinguishable

    Returns
    -------
    None.

    """
    distance in ['jensen_shannon', 'kullback_leibler']

    dif_mat = pmodel.diff(qmodel, 
                          distance = distance, 
                          annotation = False,
                          normed = normed)[0]
    
    
    #set diagonal to NANs if given same model
    if pmodel is qmodel:
        #np.fill_diagonal(dif_mat,np.nan)
        dif_mat[np.triu_indices(dif_mat.shape[0],0)] = np.nan
        
    
    Kp = pmodel.num_topics
    Kq = qmodel.num_topics
    pticks = np.arange(0,Kp,1)
    qticks = np.arange(0,Kq,1)
    
    #set-up labelling depending on distance metric
    xlabel = "%d-Topic Model" %Kq
    ylabel = "%d-Topic Model" %Kp
    if distance == "jensen_shannon":
        abbrev = "JS"
    else:
        abbrev = "KL"
        xlabel += " (q)"
        ylabel += " (p)"
        
    title = "%s-divergence of K = %d Model Topics \n from K = %d Model Topics" %(abbrev, Kp, Kq)
    
    
    #create plot
    _plot_heatmap(dif_mat, 
                 cmap = cmap,
                 figsize = figsize,
                 title = title,
                 xlabel = xlabel,
                 ylabel = ylabel)
    plt.xticks(ticks = qticks, labels = qticks, fontsize = 11)
    plt.yticks(ticks = pticks, labels = pticks, fontsize = 11)                  
    
    #optionally save figure
    if save_fig:
        figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     dpi = dpi,
                     fig_override = fig_override,
                     bbox_inches = "tight")

    
  
def get_most_sim_topics(model_main, model_other, distance, return_vals = False):
    """get topics from model2 topics that are most similar to each of model1 topics
    in terms of lowest distance metric
    
    if distance == "jensen_shannon", this is used to calculate topic similarities

    if distance == "kullback_leibler", this is used to calculate topic similarities
        and NOTE that KL divergence is always calculated so that
        the model with lower K is q 
       
        Note: the approach to choosing p and q here can make a difference in terms of
        which topics are returned, but from my experience, for the most similar pairs
        of topics, both choices of p and q tend to return the same thing 
   
    output can contain repeats
    
    optionally, also return the KL/JS divergence values
    
   
    """
    K_main = model_main.num_topics
    K_other = model_other.num_topics
    if K_main >= K_other: #this part is irrelevant for JS
        dif_mat = model_main.diff(model_other, distance = distance, annotation = False)[0]
        most_sim_topic_ids = np.argmin(dif_mat, axis = 1) #row minimums = most similar
        vals = np.min(dif_mat, axis = 1) #row minimums = most similar
        
    else:
        dif_mat = model_other.diff(model_main, distance = distance, annotation = False)[0]
        most_sim_topic_ids = np.argmin(dif_mat, axis = 0) #column minimums
        vals = np.min(dif_mat, axis = 0) #column minimums
    assert len(most_sim_topic_ids) == K_main, "Something has gone wrong - should have output length %d" % K_main
    if not return_vals:
        return(most_sim_topic_ids)
    else:
        return(most_sim_topic_ids, vals)
    
 
    
 
    
 
    
 
### TOPIC CORRELATIONS

def topic_correlations(theta_mat, plot = True, cmap = "PuOr",
               save_fig = False, fig_outpath = None, 
               fig_name = "topic_correlations_map", dpi = 200,
               fig_override = False):  
    """
    Get (pearson) correlations between topics and optionally plot a heatmap of correlations
    between topics within a model as calculated by looking at correlations between
    theta_k = theta_1k...theta_Dk vectors for all pairs of topics

    Parameters
    ----------
    theta_mat : np.array
        theta matrix for a topic model as output by get_document_matrices()
        
    plot : bool, optional
        if True, will plot heatmap. Default is True
    
    cmap : str, optional
        specify color map for heatmap. The default is "PuOr".
        
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py
        
    Returns
    -------
    full K x K matrix of topic correlations

    """
    #get # of topics
    K = theta_mat.shape[0]
        
    #get correlation matrix and remove half of it + diagonal
    df = pd.DataFrame(theta_mat.T)
    cor_mat = np.array(df.corr())
    if not plot:
        return(cor_mat)
    else:
        cor_mat_return = cor_mat.copy()
        cor_mat[np.triu_indices(cor_mat.shape[0],0)] = np.nan
        _plot_heatmap(cor_mat, cmap = cmap, xlabel = "Topics IDs", ylabel = "Topic IDs" )
        
        #tick marks
        xticks = np.arange(0,K,1)
        yticks = np.arange(0,K,1)
        plt.xticks(ticks = xticks, labels = xticks, fontsize = 11)
        plt.yticks(ticks = yticks, labels = yticks, fontsize = 11) 
        title = "Topic Correlations for %d-Topic Model" %K
        plt.title(title, size = 25, pad = 20)
        
        if save_fig:
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                         bbox_inches = "tight")
        
        return(cor_mat_return)