# -*- coding: utf-8 -*-
"""

This file contains functions for processing the output of gensim LdaModels

Specifically, it contains functions for visualizing individual topics
or grids of topics in terms of their words and those words' probabilities
or expected counts


It draws on calculations implemented in LdaOutput.py

Author: Kyla Chasalow
Last edited: August 19, 2021


"""
import numpy as np
import matplotlib.pyplot as plt
from Helpers import figure_saver
import LdaOutput
import Helpers



#function for single plot
def topic_word_barplot(word_list, topic_magnitudes, value_type,
                       overall_magnitudes = None, plot_overall_magnitudes = True,
                       title = "",  ylabel = "", 
                       set_figsize = True, figsize = (10,8), plot_title = True,
                       plot_xlabel = True, plot_ylabel = True, plot_legend = True,
                       legend_fontsize = 15, max_x = None, save_fig = False, fig_outpath = None, 
                       fig_name = "topic_word_barplot", dpi = 200,
                       fig_override = False):
    """
    Represent a single as horizontal bar plot with both topic-specific magnitudes
    and overall magnitudes represented
    
    Bars can either represent probabilties (value_type = "probability") or 
    expected counts (value_type = "counts"). Colors are preset to purple and green
    for probabilities and red and blue for counts
    
    This function is set-up as a helper function to be used in a grid
    or in some wrapper that also calculates the necessary quantities

    Parameters
    ----------
    word_list : list of strings containing words
      
    topic_magnitudes : list of floats representing either probabilities or counts
        for each word within the topic as output by get_topword_probs or
        get_topword_counts
    
    value_type : str, either "probabilities" or "counts"

    overall_magnitudes : list of floats representing either probabilities or counts
        for each word in the overall corpus as output by get_topword_probs or
        get_topword_counts
        
    
    plot_overall_magnitudes : bool, default True
        if False, only plots the topic magnitudes and overall magnitudes do not have
        to be given
        
    title : str, title of bar plot, default is empty string
        
    figsize : tuple, optional
        optionally set figure size. The default is (10,8).

    set_figsize, plot_xlabel, plot_ylabel, plot_legend, plot_title
    can be used to turn off various plot elements. defaults are True
    
    legend_fontsize : int
        controls legend size
        
    max_x : optionally set max heigh of barplot (x axis)
    

    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py


    Returns
    -------
    None.

    """
    plt.rcParams.update({'font.family':'serif'})
    
    assert value_type in ["probabilities", "counts"]
    if plot_overall_magnitudes:
        assert overall_magnitudes is not None, "if plot_overall_magnitudes = True, overall_magnitudes must not be None"
    
    
    word_list = np.flip(word_list)
    topic_magnitudes = np.flip(topic_magnitudes)
    overall_magnitudes = np.flip(overall_magnitudes)
    if set_figsize:
        plt.figure(figsize = figsize)
    
    if value_type == "probabilities": #plot overall second so that they show up on top (will tend to be smaller)
        plt.barh(word_list, topic_magnitudes, 
                 color = "mediumpurple", 
                 alpha = .8, 
                 label = "Topic probabilities",
                 edgecolor = "k")
        if plot_overall_magnitudes:
            plt.barh(word_list, overall_magnitudes, 
                     color = "lightgreen",
                     alpha = 1, 
                     label = "Corpus probabilities",
                     edgecolor = "k")
        xlab = "Probability"
        
    elif value_type == "counts": #plot overall first since tend to be bigger
        if plot_overall_magnitudes:
            plt.barh(word_list, overall_magnitudes, 
                     color = "lightblue",
                     alpha = .8, 
                     label = "Expected overall word count",
                     edgecolor='k')
        plt.barh(word_list, topic_magnitudes, 
                    color = "red", 
                    alpha = .6, 
                    label = "Expected topic word count",
                    edgecolor ='k')
        xlab = "Expected word count"
    
    if plot_legend:
        plt.legend(fontsize = legend_fontsize)
    if plot_xlabel:
        plt.xlabel(xlab, fontsize = 16)
    if plot_ylabel:
        plt.ylabel("Top %d words" % (len(word_list)), fontsize = 16)
    if max_x is not None:
        plt.xlim((0,max_x))
        
    plt.xticks(rotation = "horizontal", fontsize = 12)
    plt.yticks(fontsize = 14)
    if plot_title:
        plt.title(title, pad = 25, fontsize = 22)

        
    #optionally save figure
    if save_fig:
        figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     dpi = dpi,
                     fig_override = fig_override,
                     bbox_inches = "tight")
    


    


def topic_grid_plotter(word_lists, topic_magnitude_lists, 
                    to_plot, label_list, value_type,  
                    overall_magnitude_lists = None, plot_overall_magnitudes = True,
                    title = "", ylabel = "",
                    plot_title_and_legend = True, save_fig = False, fig_outpath = None, 
                    fig_name = "topic_barplot_grid", dpi = 200,
                    fig_override = False, display_figure = True):
    """
    Plot a grid of at most 6 topic bar plots at once.
    
    

    Parameters
    ----------
    word_lists : list of lists of strings
        as output by get_topword_probs or get_topword_counts
  
    topic_magnitude_lists : list of floats representing either probabilities or counts
        for each word within the topic as output by get_topword_probs or
        get_topword_counts
    
    overall_magnitude_lists : list of floats representing either probabilities or counts
        for each word in the overall corpus as output by get_topword_probs or
        get_topword_counts
        
        if plot_overall_magnitudes  = False, this can be None
        
    plot_overall_magnitudes : bool : default True
        if True, plots overall corpus-wide magnitudes using different-ly colored,
        overlayed bars. If False, does not and overall_magnitude_lists can be done
        
    to_plot : list of ints
        list of topic IDs. These will be used to access words and magnitudes
        from the previous three arguments and will be plotted in order given
    
    labels : list of labels of same length as to_plot
    
    value_type : str
        must be one of "counts" or "probabilities" depending on what kinds of
        magnitudes given
        
    num_topics : number of topics in the model the topics come from
        
        
    title : str, title of overall grid, default "" which amounts to no title
    
    plot_title_and_legend : bool, optional
        If False, will not plot legend and title. The default is True.
        
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

    display_figure : bool, optional, default is True
        if running function in a notebook, display_figure = True will 
        ensure figure is shown but display_figure = False will mean that
        figure is deleted after it is created (and potentially saved).
        There is little point to setting display_figure = False if save_fig
        is also false, but in case where only want to save figure, 
        setting display_figure = False can help avoid memory issues that arise
        from opening too many plots at once time

    Returns
    -------
    None.

    """
    plt.rcParams.update({'font.family':'serif'})
    
    #a few initial checks
    assert type(to_plot) == list, "to_plot must be a list"
    assert all([type(val) == int for val in to_plot]), "to_plot must be a list of ints"
    assert len(label_list) == len(to_plot), "to plot list and labels list must have same length"
    if plot_overall_magnitudes:
            assert overall_magnitude_lists is not None, "if plot_overall_magnitudes = True, overall_magnitudes must not be None"
      
     
    #figure out what kinds of magnitudes have been given
    assert value_type in ["probabilities", "counts"]
    if value_type == "probabilities":
        xlabel = "Probabilities"
    elif value_type == "counts":
        xlabel = "Expected word counts"
   
    
    #figure out what kind of grid to create and set relevant parameters so that everything shows up well
    l = len(to_plot)
    assert l <= 6, "I can only plot at most 6 plots at a time"
    
    nrow, ncol = Helpers.figure_out_grid_size(num_plots = l, num_cols = 2, max_rows = 3)
    if l == 1:
        figsize = (4, 7); top = .65; legend_loc = (.3, .79);  ylabel_loc = (-0.62, 0.4)
    elif l == 2:
        figsize = (10, 8); top = .65; legend_loc = (.42, .79); ylabel_loc = (-0.18, 0.4)
    elif l <= 4:
        figsize = (10,13); top = .82; legend_loc = (.42,.90); ylabel_loc = (-0.20, 0.5)
    elif l <= 6:
        figsize = (10,18); top = .86; legend_loc = (.43, .92); ylabel_loc = (-0.20, 0.5)
    if not plot_title_and_legend:
        if l == 2:
            figsize = (figsize[0], figsize[1] - 3)    
        else:
            figsize = (figsize[0], figsize[1] - 2)
                       
    #EXTRACT some global values
    #get max value that occurs anywhere so can set all x axes the same
    flat_list1 = [entry for elem in topic_magnitude_lists for entry in elem]     
    max_val = np.max(flat_list1)
    if plot_overall_magnitudes:
        flat_list2 = [entry for elem in overall_magnitude_lists for entry in elem]          
        max_val = np.max([[flat_list1, flat_list2]])
    
    #set up subplot
    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, sharex=True, sharey=False, figsize=figsize)
    fig.text(0.5, -0.05,  xlabel, ha='center', fontsize = 22)
    
    ylabel = ylabel
    fig.text(*ylabel_loc, ylabel,
             va='center', rotation='vertical', fontsize = 20)
    fig.tight_layout(h_pad=10, w_pad = 14)

    for i in range(ncol * nrow):
        ax = plt.subplot(nrow,ncol,i+1)
        if i < l:
            #deal with different pathways for if overall magnitudes supplied or not
            if plot_overall_magnitudes:
                overall_magnitudes = overall_magnitude_lists[to_plot[i]]
            else:
                overall_magnitudes = None     
            #pass options to individual barplot function
            topic_word_barplot(word_list = word_lists[to_plot[i]],
                           topic_magnitudes = topic_magnitude_lists[to_plot[i]],
                           overall_magnitudes = overall_magnitudes,
                           plot_overall_magnitudes = plot_overall_magnitudes,
                           title =  label_list[i],
                           value_type = value_type,
                           set_figsize = False,
                           plot_legend = False,
                           plot_title = True,
                           plot_ylabel = False,
                           plot_xlabel = False,
                           max_x = max_val*1.1)
            #get legend info
            handles, labels = ax.get_legend_handles_labels()
            
        #in case of odd number of topics requested, turn off extra grids
        if l == 3 and i == 3: 
            ax.set_visible(False)
        elif l == 5 and i == 5:
            ax.set_visible(False)

    
    #Note limitation: not allowing legend without title or title without legend because 
    #spacing gets messed up if just remove one. 
    if plot_title_and_legend:
        plt.suptitle(title, fontsize = 25)
        plt.subplots_adjust(top= top)                        

        fig.legend(handles, labels,
                   loc = legend_loc, fontsize = 15)
                    
    #optionally save figure
    if save_fig:
        figure_saver(fig_name = fig_name, 
                     outpath = fig_outpath,
                     dpi = dpi,
                     fig_override = fig_override,
                     bbox_inches = "tight")
    
    if not display_figure:
        fig.clear()
        del fig #clear it at the end - can be important when making many many figures




### MAIN WRAPPERS 
#Use the above two plot functions to plot barplots for topics in models with
# words ordered by relevance


# TO DO: IMPROVE THIS DOCUMENTATION
def topic_relevance_barplot(model, topicid, corpus, dictionary, value_type, theta_mat = None, 
                                plot_overall_magnitudes = True,
                                lamb = 0.6, topn = 20, minimum_probability = .01,
                                plot_title = True, title = None, detect_max_x = True,
                                set_figsize = True, figsize = (10,8), 
                                save_fig = False, fig_outpath = None, 
                                fig_name = "topic_barplot", dpi = 200,
                                fig_override = False):
    """
    a wrapper for topic_word_barplot that calculates the needed values and then plots 
    the barplot for the given topic
    
    if theta matrix (theta_mat) is not specified, will generate it
        

    Parameters
    ----------
    model : gensim LDA model

    topicid : int, topic to plot barplot for
        DESCRIPTION.
    corpus : corpus used to train Gensim LDA model (vector representaiton)
    
    dictionary : dictionary used to train Gensim LDA model

    value_type : str, either "probabilities" or "counts"  

    theta_mat : numpy array, optional
        K x D array as output by LdaOutput.get_doc_matrices(). The default is None.
        if not specified, will generate this
        
    plot_overall_magnitudes : Bool, optional
        If True, plots topic-specific counts/probabilities and overall corpus
        counts/(empirical) probabiltiies. If False, only plots topic-specific
        quantitites. Default is True
    
        
    lamb : float between 0 and 1, optional
        controls trade-off between word probability and "lift".
        lamb = 0 means we consider only lift (may select very rare words) 
        lamb = 1 means we consider only topic probabilities
        The default is 0.6.

    topn : int, optional
        number of top words to plot. The default is 20.
  
    minimum_probability : float, optional
        if theta_mat not specified, will use LdaOutput.get_document_matrices() 
        to getnerate it. In that case, will pass this argument on to that function.
        It is the cut-off for setting values of that matrix to 0.
        The default is .01.
        
    plot_title : bool, optional
        decide whether to plot title. The default is True.
        
    title : str, optional
        optionally specify title. The default is None.
        If None, title is set to:
            
            "Topic <topic_id> from <model.num_topics>-Topic Model \n 
            (ordered by relevance $\lambda$ = <lamb>"

        
    detect_max_x : bool, optional
        if True, detects maximum value on x axis for all topics in model
        and sets that to x limit so that x axis is scaled to make all topics in model comparable
        if you were to generate multiple of these plots. The default is True.
        
    
    set_figsize : bool, optional
        determine whether sets figure size. The default is True.
        
    figsize : tuple, optional
        figure size to sert if set_figsize = True. The default is (10,8).
 
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

    Returns
    -------
    None.

    """
    
    K = model.num_topics
    
    assert value_type in ["probabilities", "counts"], "value_type must be either 'probabilities' or 'counts'"
    assert type(topicid) == int and topicid >= 0, "topicid must be an integer >= 0"
    assert topicid < K, "topicid out of range"
    doc_lengths = LdaOutput.get_doc_lengths(corpus)
    
    if theta_mat is None:
        theta_mat = LdaOutput.get_document_matrices(model = model, 
                                     corpus = corpus, 
                                     save_theta_matrix = False, 
                                     minimum_probability = minimum_probability)
        
    
    #get relevances
    relevance_mat, order_mat = LdaOutput.get_relevance_matrix(model = model,
                                                              corpus = corpus,
                                                              dictionary = dictionary,
                                                              lamb = lamb,
                                                              phi = None)
    
    if value_type == "probabilities":
        word_lists, topic_magnitude_lists, overall_magnitude_lists = LdaOutput.get_topword_probs(order_mat = order_mat,
                                                                                          model = model, 
                                                                                          corpus = corpus,
                                                                                          dictionary = dictionary,
                                                                                          topn = topn,
                                                                                          phi = None)
    elif value_type == "counts":
        word_lists, topic_magnitude_lists, overall_magnitude_lists = LdaOutput.get_topword_counts(order_mat,
                                                                                          model = model, 
                                                                                          dictionary = dictionary, 
                                                                                          theta_mat = theta_mat,
                                                                                          doc_lengths = doc_lengths,
                                                                                          topn = topn)
        
    if detect_max_x: #get max value that occurs anywhere
        max_x = np.max([Helpers._get_max_nested_list(topic_magnitude_lists),
                       Helpers._get_max_nested_list(overall_magnitude_lists)])
    else:
        max_x = None
        
    if title is None:
        title = "Topic %d from %d-Topic Model \n" % (topicid, K)
        title = title + r"(ordered by relevance, $\lambda$ = %s)" % str(lamb)
        
    topic_word_barplot(word_list = word_lists[topicid],
                       topic_magnitudes = topic_magnitude_lists[topicid],
                       overall_magnitudes = overall_magnitude_lists[topicid],
                       title = title,
                       ylabel = "Top %d words \n" % topn,
                       value_type = value_type,
                       set_figsize= set_figsize, 
                       figsize = figsize, 
                       plot_overall_magnitudes = plot_overall_magnitudes,
                       plot_title = plot_title,
                       max_x = max_x * 1.1,
                       save_fig=save_fig,
                       fig_outpath = fig_outpath,
                       dpi = dpi ,
                       fig_name = fig_name,
                       fig_override = fig_override)
    
    
    
    
    


    


def topic_relevance_grid(model, corpus, dictionary, value_type, plot_all_topics = True,
                         custom_list = None, theta_mat = None, lamb = 0.6, topn = 20,
                         minimum_probability = .01, plot_overall_magnitudes = True,
                         plot_title_and_legend = True, first_title_only = False, custom_title = None,
                         save_all_plots = False, custom_name = None, dpi = 200, fig_outpath = None,
                         fig_override = False, display_figure = True):
    """
    Wrapper that relies on many of the functions above.
    
    Plots all topics in given model in grids of 6 at a time, possibly with a smaller remainder
        grid if the number of topics is not evenly divisilbe by 6
    
    Optionally, specify a custom list of only some topics to plot and it will again
    plot all topics given in grids of 6 at a time with remainder if needed
        
    Parameters
    ----------
    
    model : gensim LDA model

    corpus : corpus used to train Gensim LDA model (vector representaiton)
    
    dictionary : dictionary used to train Gensim LDA model

    value_type : str, either "probabilities" or "counts"
        
    plot_all_topics: bool, default True
        if True, figures out how many grids needed to create plots for all topics
        in the model
        
        if False, requires custom_list to be specified and figures out how many grids
        needed to create plots for all topics in the list
    
    theta_mat : numpy array, optional
        array output by get_document_matrices() The default is None.
        If None, will calculate this using model and corpus but NOTE
        that there is some randomness in this, meaning different calls to the
        function are not identical. It may be better to run
        get_document_matrices() outside of this function with save option turned on,
        save it, load it, and then use it in this function
        
    topn : int, optional
        number of top words to plot. The default is 20.
        
    lamb : float between 0 and 1, optional
        controls trade-off between word probability and "lift".
        lamb = 0 means we consider only lift (may select very rare words) 
        lamb = 1 means we consider only topic probabilities
        The default is 0.6.

        
    minimum_probability : float, optional
        cut-off to use when calculating theta_matrix if theta_matrix
        argument is None. The default is .01.
        
    plot_title_and_legend : Bool, optional
        If True, plots title and legend. 
        If False, does not on any plot 
        
    plot_overall_magnitudes : Bool, optional
        If True, plots topic-specific counts/probabilities and overall corpus
        counts/(empirical) probabiltiies. If False, only plots topic-specific
        quantitites. Default is True
        
    first_title_only : Bool, optional
        If True (and plot_title_and_legend is true), 
        only plots title and legend for first plot in the set of grids
        The default is False.
        
    custom_title : Str, optional
        optionally specify a custom title but warning: this same title will be the same
        for every grid (if there are multiple grids)
        
    save_all_plots : bool, optional
        if True, saves all figures. The default is False.
        
        if custom_name is None:
            automatically names the figures as Ktopic_grid_# where K is the 
            number of topics and #'s are in order
        else
            uses custom_name and appends # at end to make each grid
        
    dpi and fig_outpath are standard arguments to fig_saver helper function


    display_figure : bool, optional, default is True
        if running function in a notebook, display_figure = True will 
        ensure figures are all shown but display_figure = False will mean that
        figures are deleted after they are created (and potentially saved).
        There is little point to setting display_figure = False if save_all_plots
        is also False, but in case where only want to save figures, 
        setting display_figure = False can help avoid memory issues that arise
        from opening too many plots at once time

    Returns
    -------
    None.

    """
    
    num_topics = model.num_topics
    
    assert value_type in ["probabilities", "counts"], "value_type must be either 'probabilities' or 'counts'"
    if not plot_all_topics:
        assert custom_list is not None, "if plot_all_topics = False, custom_list of topics must be specified"
        assert type(custom_list) == list, "custom_list must be list"
        assert all(type(i) == int and i >= 0 and i < num_topics for i in custom_list), "custom_list must contain positive integers less than num_topics"
    
    
    doc_lengths = LdaOutput.get_doc_lengths(corpus)
    if theta_mat is None:
        theta_mat = LdaOutput.get_document_matrices(model = model, 
                                     corpus = corpus, 
                                     save_theta_matrix = False, 
                                     minimum_probability = minimum_probability)
        
    
    #get relevances
    relevance_mat, order_mat = LdaOutput.get_relevance_matrix(model = model,
                                                              corpus = corpus,
                                                              dictionary = dictionary, 
                                                              lamb = lamb,
                                                              phi = None)
    
    if value_type == "probabilities":
        word_lists, topic_magnitude_lists, overall_magnitude_lists = LdaOutput.get_topword_probs(order_mat = order_mat,
                                                                                          model = model, 
                                                                                          corpus = corpus,
                                                                                          dictionary = dictionary,
                                                                                          topn = topn,
                                                                                          phi = None)
    elif value_type == "counts":
        word_lists, topic_magnitude_lists, overall_magnitude_lists = LdaOutput.get_topword_counts(order_mat,
                                                                                          model = model, 
                                                                                          dictionary = dictionary, 
                                                                                          theta_mat = theta_mat,
                                                                                          doc_lengths = doc_lengths,
                                                                                          topn = topn)
    #figure out how many grids to create    
    if plot_all_topics:
            custom_list = list(range(model.num_topics))
    to_plot_list = Helpers.get_grids(to_plot_list = custom_list,
                                         num_col = 2, num_row = 3)

    
    #set title and y label 
    if custom_title is None:
        title = "Most relevant words for %d-Topic Model Topics"% (num_topics) # + topics_string[:len(topics_string)-1] #remove last ","
    else:
        title = custom_title
    ylabel = r'Top %d words by relevance ($\lambda$ = %s)' % (topn, str(lamb))
    
    #generate plots
    for i, plot_list in enumerate(to_plot_list):
        
        #handle save name
        if custom_name is not None:
            fig_name = custom_name + str(i)
        else:
            if plot_all_topics:
                fig_name = "%d_topic_grid" % num_topics + str(i)        
            else:   
                #name file using string of #'s of topic IDs included
                string_ID = "".join([str(k) for k in plot_list])
                fig_name = "%d_topic_grid" %num_topics + str(i) + string_ID
        
        if plot_title_and_legend and first_title_only and i != 0: #anything after first
            plot_title_and_legend = False
        topic_grid_plotter(word_lists = word_lists,
                                  topic_magnitude_lists = topic_magnitude_lists,
                                  overall_magnitude_lists = overall_magnitude_lists,
                                  to_plot = plot_list, 
                                  label_list = ["Topic " + str(num) for num in plot_list],
                                  value_type = value_type,
                                  title = title,
                                  ylabel = ylabel,
                                  plot_title_and_legend = plot_title_and_legend,
                                  plot_overall_magnitudes = plot_overall_magnitudes,
                                  save_fig = save_all_plots, 
                                  fig_name = fig_name,
                                  fig_outpath = fig_outpath,
                                  dpi = dpi,
                                  fig_override = fig_override,
                                  display_figure = display_figure)
    
    
    
    
    
    
    
