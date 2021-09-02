# -*- coding: utf-8 -*-
"""

This file is for visualizing LDA output

In particular, it creates plots for comparing per-topic metrics across models

Author: Kyla Chasalow
Last edited: August 1, 2021
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from gensim.models.coherencemodel import CoherenceModel


import Scatterbox
import LdaOutput
from Helpers import figure_saver



def visualize_spread(model_summaries, metric, labels,  
                               topn_coherence = None, topn_phf = None, thresh = None,
                               color_box = "purple", color_point = "blue", xlabel = " ",sup_title = " ",
                               plot_points = True,  alpha_box = .3, alpha_point = .3,
                               save_fig = False, fig_outpath = None, 
                               fig_name = "topicmetricspread", dpi = 200,
                               fig_override = False):
    
    """
    Visualize the spread of the values of given metric for the topics in each model 
    using boxplots and optionally, points. Use this to get a more detailed sense of how models 
    compare in terms of metric (e.g. rather than just comparing averages)
    
    If metric = 'all', plots all four metrics on a grid. Note that in this case, colors have been
    pre-set so color_box, color_point, alpha_box, and alpha_point have no effect

    Parameters
    ----------
    
    model_summaries : list of arrays as output by LdaOutput.topic_summarizer() in LdaOutput.py
        
    metric : str
        must be one of ["coherence","phf","entropy", "kl", 'all']
      
    labels : list of strings
        list of labels for each boxplot. Length should 
        be the same length as model_summaries
    
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
     
    color_box, color_point : str, optional
        boxplot color for boxplot and points respectively
        The defaults are "purple" and "blue"
        
    alpha_box, alpha_point : float, optional
        transparency value for boxplot and points respectively
        The default is .3
    
    plot_points : Bool, optional
        optionally also plot points on the boxplots for each model
        
    xlabel : str, optional
        x axis label, default is None, left as optional parameter
        because might use function to make multiple kinds of comparisons
        
        in case of metric = "all", this is x label for overall grid
        
    sup_title : str, optional
        only relevant if metric = "all". In this case, this is the overall title for the grid

    
    Returns
    -------
    None

    """
    assert len(model_summaries) == len(labels), "labels must contain same number of elements as summary_array"
    assert metric in ["coherence","phf","entropy", "kl", 'all'], "metric must be one of\
        ['coherence','phf','entropy', 'kl', 'all']"
    
    #assert len(model_list) == len(labels), "labels must contain same number of elements as model_list"
    
    
    
    if metric != "all":
        #GENERAL SETTINGS
        ylim = None
        ylabel = "Topic " + metric
        title = "Spread of " + metric.capitalize() + " for Topics in each Model" 
       
        #SOME CUSTOMIZING BY TYPE OF PLOT
        #label used in title
        if metric == "phf":
            assert thresh is not None, "thresh must be specified for phf"
            assert topn_phf is not None, "topn_phf must be specified for phf"
            ylabel = "percentage"
            ylim = (-0.05,1.05)
            title = "Percentage of High-Frequency (top %s" %  str(100*thresh) 
            title += "% of corpus) Words" + " \n Among Top %d Words in each Topic" % topn_phf 
       
        elif metric == "kl":
             title = "Spread of KL-Divergence for Topics in each Model"          
             ylabel = "KL-Divergence from Corpus Distribution"
        
        elif metric == "coherence":
             assert topn_coherence is not None, "topn_coherence must be specified for coherence"
             title += "\n (using top %d words)" % topn_coherence    
        
        
        Scatterbox.scatterbox(model_summaries, labels = labels, 
                              plot_points = plot_points, title = title,
                              color_box = color_box, color_point = color_point,
                              xlabel = xlabel, ylabel = ylabel, ylim = ylim,
                              alpha_box = alpha_box, alpha_point = alpha_point,
                              legend_point_label = "Topics")
        
        if save_fig:
               figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                         bbox_inches = "tight")
        
    
    elif metric == "all":
        assert thresh is not None, "thresh must be specified for when plotting all metrics"
        assert topn_phf is not None, "topn_phf must be specified when plotting all metrics"
        assert topn_coherence is not None, "topn_coherence must be specified when plotting all metrics"
            
            
        coherences = [elem[0] for elem in model_summaries]
        entropies = [elem[1] for elem in model_summaries]
        kl = [elem[2] for elem in model_summaries]
        phf = [elem[3] for elem in model_summaries]
        
        
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False, figsize=(20, 20))
        fig.text(0.5, 0.04,  xlabel, ha='center', fontsize = 24)
        fig.text(0.04, 0.5, 'Per-Topic Metric', va='center', rotation='vertical', fontsize = 24)
        
        plt.subplot(2,2,1)
        Scatterbox.scatterbox(arr_list = coherences, labels = labels,
                              plot_points = plot_points,
                              xlabel = " ", ylabel = "Coherence",
                              title = "Spread of Coherence for Topics in each Model \n (using top %d words)" % topn_coherence,
                              color_box = "purple", color_point = "blue",
                              alpha_point = .5, alpha_box = .3, set_fig_size = False,
                              plot_legend = False)
        
        plt.subplot(2,2,2)  #only plot legend for one of them
        Scatterbox.scatterbox(arr_list = entropies, labels = labels,
                              plot_points = plot_points,
                              xlabel = " ", ylabel = "Entropy",
                              title = "Spread of Entropy for Topics in each Model",
                              color_box = "lightgreen", color_point = "blue",
                              alpha_point = .5, alpha_box = .4, set_fig_size = False,
                              plot_legend = True,  legend_point_label = "Topics")
  
        plt.subplot(2,2,3)
        Scatterbox.scatterbox(arr_list = kl, labels = labels,
                              plot_points = plot_points,
                              xlabel = " ", ylabel = "KL-Divergence from Corpus Distribution",
                              title = "Spread of KL-Divergence for Topics in each Model",
                              color_box = "lightblue", color_point = "blue",
                              alpha_point = .5, alpha_box = .4, set_fig_size = False,
                              plot_legend = False)


        plt.subplot(2,2,4)
        
        phf_title = "Percent of words from top " +  str(100*thresh) + "% of corpus \n among top " + str(topn_phf)+" words in each topic" 
       
        Scatterbox.scatterbox(arr_list = phf, labels = labels,
                              plot_points = plot_points,
                              xlabel = " ", ylabel = "Percentage High Frequency",
                              title = phf_title, 
                              color_box = "orange", color_point = "blue", ylim = (-0.05,1.05),
                              alpha_point = .5, alpha_box = .3, set_fig_size = False,
                              plot_legend = False)
        
           
        
        plt.suptitle(sup_title, fontsize = 30)
        if save_fig:
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                         bbox_inches = "tight")




     
        
        



def multi_model_pairplot(model_summaries, labels, colors,
                         topn_coherence = 10, topn_phf = 25, thresh = 0.01,
                         alpha = .55, save_fig = False, fig_outpath = None, 
                         fig_name = "topicmetricpairplot", dpi = 200,
                         fig_override = False):
    """
    create scatterplot matrix for comparing all the topic metrics from topic_summarizer()
    to each other AND across all models (points are color-coded by model)
    used to investigate questions like: are more coherent topics also higher in entropy?
 
    Parameters
    ----------
    model_summaries : list of arrays as output by topic_summarizer() in LdaOutput.py
         with metric == "all"
         
    labels : list of strings
        labels for the box plots
        must be of same length as model_list
        
    colors : list with color for each model 
            
    topn_coherence : int, optional, default 10
        # of words to consider when calculating coherence
        
    topn_phf : int, optional, default = 25
        # of words to consider when calculating phf
    
    thresh : int in (0,1), optional
        threshold for phf. The default is 0.01.
   
    alpha : float, optional
        transparency of points in scatterplots. The default is .55.
        
    colors : list, optional
        optionally, specify list of colors of same length as labels to use
        in coloring each entry of model_list, default is None and in that case
        will use a standard option

    Returns
    -------
    The underlying :class:`PairGrid` instance from sns.pairplot() for further tweaking.
    """
    
    
    assert len(model_summaries) == len(labels), "labels must contain same number of elements as model_summaries"
    if colors is not None:
        assert len(colors) == len(labels), "colors must be same length as labels"
    
    # out = LdaOutput.topic_summarizer(model_list, "all", corpus, dictionary, 
    #                  topn_coherence =topn_coherence, 
    #                  topn_phf = topn_phf, thresh = thresh)

    #create labels column
    models = [np.repeat(name, elem[0].shape[0]) for (elem,name) in zip(model_summaries,labels)]
    models = np.concatenate(models)
    
    df_list = [pd.DataFrame(elem.T) for elem in model_summaries]
    for d in df_list:
        d.rename(columns = {0:'Coherence', 1:'Entropy', 2:"KL_Divergence", 3:"PHF"}, inplace = True) 
    
    df = pd.concat(df_list, ignore_index = True)
    df = pd.concat([df,pd.DataFrame(models)], axis = 1)
    df.rename(columns = {0:'Models'}, inplace = True)
    
    
    sns.set_context("notebook", rc={"axes.labelsize":18},font_scale=1.4)
    #reordering to avoid red and green or orange and yellow as long as possible
    grid = sns.pairplot(df, corner = True, hue = "Models",
                        palette = colors, #another option is 'colorblind'
                        plot_kws={'alpha':alpha})
    grid.fig.suptitle("Relationships Between Topic Metrics")
     
    
    #optionally save figure
    if save_fig:
        figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     dpi = dpi,
                     fig_override = fig_override)
    
    
    return(grid)

     















def topic_metric_order_plot(model, corpus, dictionary, metric,
                               figsize = (8,10), 
                               topn_coherence = 10, topn_phf = 25, thresh = 0.01,
                               save_fig = False, fig_outpath = None, 
                               fig_name = "topic_order_plot", dpi = 200,
                               fig_override = False):
    """
    
    plot topic metrics vertically in order from greatest (at top) to least (at bottom)
        * y axis is topic IDs in order
        * x axis is metric
        * each topic's metric is represented by a point
        
    Idea is to easily be able to see both the distribution of topic metrics and
    which particular topics are larger or smaller 
    

    Parameters
    ----------
    model : gensim LDA model
    
    corpus : gensim corpus
        corpus used to train LDA models
        (vector-count representation)
             
    dictionary : gensim dictinary 
        dictionary used to train LDA models

    metric : str
        one of "coherence","entropy", "kl","phf"

    figsize : tuple, optional

    topn_coherence : int, optional
        Number of top words to use when calculating coherence. The default is 10.
    
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

    Returns
    -------
    None.

    """
    plt.rcParams.update({'font.family':'serif'})
   
    possibilities = ["coherence","entropy","kl","phf"]
    assert metric in possibilities, "metric must be one of " + str(possibilities)
    
    if metric == "kl":
        metric_label = "KL-divergence from Corpus \n"
    elif metric == "phf":
        metric_label = "PHF"
    else:
        metric_label = metric.capitalize()
    
    out = np.array(LdaOutput.topic_summarizer([model], 
                                              metric = metric, 
                                              corpus = corpus,
                                              dictionary = dictionary, 
                                              topn_phf = topn_phf,
                                              thresh = thresh,
                                              topn_coherence = topn_coherence)[0])
    #get values
    K = model.num_topics
    yvals = list(range(K))
    order = np.argsort(out)
    values = out[order]
    
    #plotting 
    plt.figure(figsize = figsize)
    plt.grid(b = True, axis = "both", alpha = 0.25)
    a = plt.scatter(values, yvals, s = 80, edgecolor = "blue")
    a.set_zorder(2) #grid behind points
    plt.yticks(ticks = yvals, labels = order, fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.ylabel("Topic ID", fontsize = 20)
    
    #adjust x label depending on kind of metric
    xlab = "Topic %s" % (metric_label)
    if metric == "coherence":
        xlab += "\n(using top %d words)" % topn_coherence
    elif metric == "phf":
        xlab += "\n(for top %d words, threshold = %s)" % (topn_phf, str(thresh))
    plt.xlabel(xlab, fontsize = 20)
    
    plt.title("Topic %s by Topic for %d-Topic Model" % (metric_label, K), pad = 20, fontsize = 20)

    if save_fig:
         figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                        )   
    return(out)











### WARNING: The following function requires access to the actual models    
# TO DO: CHANGE SO CAN ALSO LOOK AT MEDIAN COHERENCE?


def evaluate_coherence_sensitivity(model_list, labels, corpus, dictionary, topn_range = (5,20),
                                   title_annotation = "", save_fig = False, fig_outpath = None, 
                              fig_name = "coherencesensitivity", dpi = 200,
                              fig_override = False):
    """
    
    Calculate mean u_mass coherence for each model in model list while 
    varying the number of top words (topn) used to calculate coherence. Then plot scatterplot
    with mean coherence on y axis, topn values on x axis, and points 
    color coded by model

    Parameters
    ----------
    model_list : list of gensim LDA models
        
    labels : labels for each model to include in plot legend

    topn_range : tuple of length 2 or 3, default (5,20)
        range of values of topn to examine, with optional third element of tuple
        specifying a stepsize
        (a,b) means will examine a,a+1,a+2...b-1
        (a,b,5) means will examine a, a+5, a+10...b (if b evenly divisible by 5) 
    
    corpus : gensim corpus
        corpus used to train LDA models
        (vector-count representation)
             
    dictionary : gensim dictinary 
        dictionary used to train LDA models
        
    title_annotation : str
        optionally add an annotation to the title on a line below the 
        standard title

    Returns
    -------
    list of lists containing mean coherences for each value of topn for
    each model

    """
    
    assert len(model_list) == len(labels), "labels must have same length as model list"
    assert type(topn_range) == tuple, "topn_range must be tuple of length 2 or 3"
    assert len(topn_range) == 3 or len(topn_range) == 2, "topn_range must be tuple of length 2 or 3"

    if len(topn_range) ==2:
        x = np.arange(topn_range[0], topn_range[1], 1)
    
    elif len(topn_range) == 3:
        x = np.arange(*topn_range)                                                       
    
    coherence_list =  [[CoherenceModel(m, 
                                   corpus=corpus, 
                                   dictionary=dictionary,
                                   coherence='u_mass',
                                   topn = n).get_coherence() for n in x] for m in model_list]
   
    plt.rcParams.update({'font.family':'serif'})
    plt.figure(figsize = (14,8))
    for i,elem in enumerate(coherence_list):
        plt.scatter(x, elem, label = labels[i], s = 50)
        plt.plot(x, elem, linewidth = 2)
        
    plt.title("Sensitivity of coherence metric to number of top words (M) used to evaluate it" + "\n" + title_annotation,
              pad = 20,
              fontsize = 20)
    plt.xlabel("M", fontsize = 18)
    plt.ylabel("Mean Coherence", fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.legend(fontsize = 15)
 
    if save_fig:
               figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                        )   
 
    return(coherence_list)


