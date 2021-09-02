# -*- coding: utf-8 -*-
"""

This file contains functions for processing the output of gensim LdaModels

Specifically, it contains functions for visualizing topic size calculations

It draws on calculations implemented in LdaOutput.py 

Author: Kyla Chasalow
Last edited: July 31, 2021


"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Helpers import figure_saver
import LdaOutput




def topic_size_comparison_plot(word_counts = None, doc_counts = None, means = None, medians = None,
                         figsize = (10,7), plot_type = "barplot", nbins = 10,
                         plot_title = True, custom_title = None,
                          save_fig = False, fig_outpath = None, 
                          fig_name = "sizecomparisonplot", dpi = 200,
                          fig_override = False):
    """

    plot either a bar plot or a pairwise scatterplot comparing the topic size metrics for each topic in a model.
    function must be given at least one of word_counts, doc_counts, means, or medians arguments
    but it is not required to supply all of them
    
    bar plot is not recommended for large number of topics as it can get cluttered.

    Parameters
    ----------
    word_counts  np.array, optional
        list of topic word counts as output by get_topic_word_counts. The default is None.
        
    doc_counts : np.array, optional
       list of topic document counts as output by get_topic_doc_counts. The default is None.
        
    means : np.array, optional
        list of mean theta for each topic as output by get_topic_means. The default is None.
        
    medians : np.array, optional
        list of median theta for each topic as output by get_topic_medians . The default is None.
        
    plottype : str, must be one of "scatterplot" or "boxplot"
      
    nbins : only used of plottype = "scatterplot". Controls number of bins for
        histograms on diagonal of the scatterplot grid
        
    figsize : tuple, optional
        set figure size. The default is (10,7).
        
    plot_title : turn off title - e.g. if want to show both scatterplot and boxplot and don't
    want to show title twice
    
    custom_title : optionally, specify a custom title. Otherwise, default title is "Topic Size"
        if only plotting one measure and "Comparing Measures of Topic Size" if plotting multiple
        
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

    Returns
    -------
    None.

    """
    assert word_counts is not None or doc_counts is not None or means is not None or medians is not None, "at least\
        one of word_counts, doc_counts, means, or medians must not be None for this function to have any point"
    assert plot_type in ["barplot", "scatterplot"], "plot_type must be barplot or scatterplot"
      
    # set up dictionary for bar plot labels
    label_dict = {}
    label_dict[-1] = "filler" #so that np.max works below for first addition to this dictionary
    
    #list to hold bar plot bar heights for each topic size metric given
    dataframe_list = []
    num_given = 0
    
    #figure out which are given and add to label dictionary accordingly
    if word_counts is not None:
        label_dict[np.max(list(label_dict.keys()))+1] = "Word Count"
        dataframe_list.append(word_counts/np.sum(word_counts))
        K = len(word_counts)
        num_given +=1
    if doc_counts is not None:
        label_dict[np.max(list(label_dict.keys()))+1] = "Doc Count"
        dataframe_list.append(doc_counts/np.sum(doc_counts))
        K = len(doc_counts)
        num_given +=1
    if means is not None:
        label_dict[np.max(list(label_dict.keys()))+1] = "Mean"
        dataframe_list.append(means/np.sum(means))
        K = len(means)
        num_given +=1
    if medians is not None:
        label_dict[np.max(list(label_dict.keys()))+1] = "Median"
        dataframe_list.append(medians/np.sum(medians))
        K = len(medians)
        num_given += 1
    
    #create data frame, adding on a topic label column
    df = pd.DataFrame(dataframe_list + [range(K)]).T
    label_dict[np.max(list(label_dict.keys()))+1] = "Topic" #add topic on last
    df = df.rename(label_dict, axis = 1)
    
    #delete filler label added at the start to make np.max method of determining
    #label_dict keys work for the first addition to it
    del label_dict[-1]
    
    plt.rcParams.update({'font.family':'serif'})
    
    if plot_type == "barplot":
        #bar plot - figsize can be adjusted by user but colors and alpha are pre-set
        plt.rcParams.update({'font.family':'serif'})
        df.plot(x="Topic",
                y=list(label_dict.values())[:num_given], 
                kind="bar",
                figsize=figsize,
                color = ["brown","orange", "blue","green"][:num_given],
                alpha = .55,
                edgecolor = "k")
        
        #further plot aesthetics
        plt.xlabel("Topic", fontsize = 18)
        plt.ylabel("Normalized Measure of Topic Size", fontsize = 18)
        if plot_title:
            if custom_title is None:
                if df.shape[1] == 2: #only one measure of topic size plotted
                    custom_title = "Topic Size"
                else:
                    custom_title = "Comparing Measures of Topic Size"
            plt.title(custom_title, pad = 15, fontsize = 25)
            
        plt.xticks(range(K),[str(int(i)) for i in range(K)],fontsize = 14, rotation = 0) #show topics as #
        plt.yticks(fontsize = 14)
        plt.legend(fontsize = 18)

    elif plot_type == "scatterplot":
        
        #get maximum value 
        max_val = df.max()[:(df.shape[1]-1)].max()
        
        sns.set_context("notebook", rc={"axes.labelsize":18},font_scale=1)
        g = sns.pairplot(df.iloc[:,:df.shape[1]-1], corner = True,
                         plot_kws={"s": 75, "color": "green", "alpha" : 0.7, "edgecolor":"lightgreen"}, 
                         diag_kind = "hist", diag_kws = {"color" : "green", "alpha" : 0.1, 
                                                         "binwidth":max_val/nbins,#overrides bins argument
                                                         "binrange":(0,max_val)})
            #note: sns.pairplot uses sns.histplot so need arguments for that function in diag_kws
        if plot_title:
            g.fig.suptitle("Comparing Measures of Topic Size", y=1.02, fontsize = 22) # y= some height>1
        #set limits to start at 0
        for ax in g.axes.flatten():
            if ax is not None:
                 ax.set_ylim(bottom = 0, top = max_val * 1.1)
                 ax.set_xlim(left = 0, right = max_val * 1.1)
                 ax.tick_params(rotation = 40) #improve visibility
           
    
    
    #optionally save figure
    if save_fig:
        figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     dpi = dpi,
                     fig_override = fig_override,
                     bbox_inches = "tight")
    
    return(df)
    



def alpha_comparison_plot(model, sizes, label, color = "black", alpha = 1.0, figsize = (8,5),
                          save_fig = False, fig_outpath = None, 
                          fig_name = "alphasizeplot", dpi = 200,
                          fig_override = False):
    """
    
    plots a scatterplot comparing values of alpha hyperparameters from LDA model
    to topic sizes and includes x=y line to aid comparison. Would expect to
    see alignment, though not perfectly. This function is more a sanity check
    type function - it isn't surprising if size and alpha value are correlated
    but it is worrying if they are wildly uncorrelated...then perhaps something has
    gone wrong
    
    Parameters
    ----------
    model : Gensim LDA model

    sizes : list 
        list of topic sizes (word counts, doc counts, means, medians)
        must be same length as model.alpha
    
    label : str
        label to be used in title and y axis to describe what kind of 
        topic size measure this is. Fills in "Topic ____ vs alpha values""

    color : str, optional
        color of scatterplot. The default is "black".
        
    figsize : tuple, optional
        set figure size. The default is (8,5).

    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py


    Returns
    -------
    None.

    """
       
    #get alphas and check if alphas are asymmetric
    #still meaningful if symmetric but in that case, would expect sizes to all
    #be about the same
    plt.figure(figsize = figsize)
    alpha_vals = model.alpha #note: different from the plotting transparency parameter!
    if all([elem == alpha_vals[0] for elem in alpha_vals]): #if all the same
        print("Note: it looks like alpha was set to be symmetric")
    assert len(alpha_vals) == len(sizes), "number of topics mismatch: sizes is not the same length as model.alpha"

    sizes = sizes / np.sum(sizes) 
    
    #plot sizes vs alpha values
    plt.scatter(alpha_vals, sizes, s = 50, color = color, alpha = alpha)
    max_val = np.max([alpha_vals, sizes])
    plt.plot(np.linspace(0,max_val*1.1,100),np.linspace(0,max_val*1.1,100), 
            color = "grey", alpha = .6, 
             dashes=(1.1, 5.),
             dash_capstyle = "round",
            linewidth = 2)
    plt.text(max_val * .95, max_val * .99, "x=y", fontsize = 15, color = "grey")
    plt.ylim(0,)
    plt.xlim(0,)
    plt.title(r"Topic %s vs $\alpha$ values" % label,
             pad = 15,
             fontsize = 20)
    plt.xlabel(r"$\alpha$", fontsize = 16)
    plt.ylabel("Topic %s (normalized)" % label, fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    
    #optionally save figure
    if save_fig:
        figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     dpi = dpi,
                     fig_override = fig_override)
    
    
    

def metric_size_comparison(model, metric, corpus, dictionary, sizes, label,
                           topn_coherence = 10, topn_phf = 25, thresh = 0.01,
                           figsize = (10,6), set_figsize = True,
                           plot_title = True, plot_ylabel = True, plot_xlabel = True,
                           save_fig = False, fig_outpath = None, 
                           fig_name = "metricsizeplot", dpi = 200,
                           fig_override = False):
    """
    
    plot scatterplot to examine relationship between metric and topic size calculation
    (as returned by get_topic_sizes or get_topic_counts or get_topic_means or
     get_topic_medians, though the last one is not recommended)

    Parameters
    ----------
    model : Gensim LDA model
    
    metric : str, one of ["coherence","entropy","kl","phf"]
    
    sizes : list 
        list of topic sizes (word counts, doc counts, means, medians)
    
    label : str
        label to be used in title and y axis to describe what kind of 
        topic size measure this is. Fills in "Topic <metric> vs Topic ____ "

    corpus : corpus used to train Gensim LDA model (vector representaiton)
    
    dictionary : dictionary used to train Gensim LDA model
    

    topn_coherence : int, optional
        number of top words to use when calculating coherenced. The default is 10.
    topn_phf : int, optional
        number of top words to use when calculating percent high frequency. The default is 25.
    thresh : float, optional
        threshold for calculating percent high frequency. The default is 0.01.
        
    figsize : tuple, optional
        figure size. The default is (10,6).
       
    set_figsize, plot_title, plot_ylabel, plot_xlabel are bool, default True,
        which can be used to optionally turn off various plot features

    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

    Returns
    -------
    None.

    """
    #normalize
    sizes = sizes / np.sum(sizes)
    
    #obtain values of metric
    metric_vals = LdaOutput.topic_summarizer([model], 
                                        metric = metric,
                                        corpus = corpus,
                                        dictionary = dictionary, 
                                        topn_coherence = topn_coherence, 
                                        topn_phf = topn_phf,
                                        thresh = thresh)
    
    #Handle metric-specific plot elements
    if metric == "kl":
        title_metric = "KL-divergence"
        xlab = "Topic KL-divergence from corpus frequencies"
        color = "blue"
        alpha = .4
    elif metric =="coherence":
        title_metric = "Coherence"
        xlab = "Topic coherence (using top %d words)" % topn_coherence
        color = "Purple"
        alpha = .6
    elif metric == "entropy":
        title_metric = "Entropy"
        xlab = "Topic entropy"
        color = "green"
        alpha = .4
    elif metric == "phf":
        title_metric = "PHF"
        xlab = "Percent highest frequency words (top " + str(thresh*100) + "%)"
        color = "orange"
        alpha = .8
    
    plt.rcParams.update({'font.family':'serif'})
            
    #optionally set figure size 
    if set_figsize:
        plt.figure(figsize = figsize)
        
    #plot the metric vs the sizes
    plt.scatter(metric_vals, sizes, 
                s = 95, 
                color = color, alpha = alpha)
    
    
    #plot aesthetics
    plt.ylim(0,)
    if plot_title:
        plt.title("Topic %s vs Topic %s" % (title_metric, label),
                 pad = 15, 
                 fontsize = 22)
    if plot_xlabel:
        plt.xlabel(xlab,
                  fontsize = 16)
    if plot_ylabel:
        plt.ylabel("Topic %s (normalized)" % (label), fontsize = 16)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    
    #optionally save figure
    if save_fig:
        figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     dpi = dpi,
                     fig_override = fig_override,
                     bbox_inches = "tight")
    
    
   

def metric_size_comparison_grid(model, corpus, dictionary, sizes, label,
                                topn_coherence = 10, topn_phf = 25,
                               thresh = 0.01, save_fig = False, fig_outpath = None, 
                              fig_name = "metricsizegrid", dpi = 200,
                              fig_override = False):
    """
    
    plot grid of scatterplots for all four metric options in metric_size_comparison()
    
    Parameters
    ----------
    model : Gensim LDA model

    corpus : corpus used to train Gensim LDA model (vector representaiton)
    
    dictionary : dictionary used to train Gensim LDA model
    
    sizes : list 
        list of topic sizes (word counts, doc counts, means, medians)
        must be same length as model.alpha
    
    label : str
        label to be used in title and y axis to describe what kind of 
        topic size measure this is. Fills in "Topic ____ vs alpha values""
        
    topn_coherence : int, optional
        number of top words to use when calculating coherenced. The default is 10.
    topn_phf : int, optional
        number of top words to use when calculating percent high frequency. The default is 25.
    thresh : float, optional
        threshold for calculating percent high frequency. The default is 0.01.
    
      
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

    Returns
    -------
    None.

    None.

    """
    
    metrics = ["coherence","entropy","kl","phf"]
    
    
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(15, 15))
    fig.text(0.5, -0.06,  "Per-Topic Metrics", ha='center', fontsize = 24)
    fig.text(-0.06, 0.5, 'Topic %s (normalized)' % (label), va='center', rotation='vertical', fontsize = 24)
    fig.tight_layout(h_pad=8, w_pad = 7)
    for i in range(4):
        plt.subplot(2,2,i+1)
        metric_size_comparison(model = model,
                              sizes = sizes,
                              label = label,
                              metric = metrics[i],
                              corpus = corpus,
                              dictionary = dictionary,
                              topn_coherence = topn_coherence,
                              topn_phf = topn_phf,
                              thresh = thresh,
                              plot_title = False,
                              plot_ylabel = False,
                              set_figsize = False,  #if True, messes up grid
                              save_fig = False)
    suptitle = "Comparing Topic Metrics to Topic %s" % label
    plt.suptitle(suptitle, fontsize = 27)
    plt.subplots_adjust(top=.92)                        
         
    #optionally save figure
    if save_fig:
        figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     dpi = dpi,
                     fig_override = fig_override,
                     bbox_inches = "tight")
    
    
    

