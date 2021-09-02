# -*- coding: utf-8 -*-
"""

This file contains functions for processing the output of gensim LdaModels

Specifically, it contains functions for visualizing topic size over time
and how word use within topics changes over time

It draws on calculations implemented in LdaOutput.py 
and on plot from LdaOutputWordPlots.py

Author: Kyla Chasalow
Last edited: August 29, 2021


"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import Helpers
from Helpers import figure_saver
import LdaOutput
import LdaOutputWordPlots
import LdaOutputTopicSimilarity


#######TOPIC SIZE OVER TIME 


## Plot a Single Topic

def _topic_sizes_by_year_plotter(years, per_topic_vals, topic_id, max_y_val = None,
                                plot_xlabel = True, plot_ylabel = True, plot_title = True,
                                title = None,
                                set_figsize = True, figsize = (10,5),
                                left_zoom = 0, right_zoom = 0, x_tick_rotation = 0):
    """
    Helper for plotting topic over time plot
    
    
    Parameters
    ----------
    years : list of years
 
    per_topic_vals : list of topic sizes by year    
    
    topic_id : the id of the topic to plot - only used for labeling purposes
        
    
    max_y_val : float or int, optional
        optionally set upper limit of y axis. The default is None.
        lower limit is set to 0
        
    plot_xlabel... set_figsize turn on and off plot elements
        default for all is True but might need to set to false
        if plotting on a grid
        
    title : str, optional
        if None, automatically plots "Topic <topic_id>" as title
         
    figsize : tuple, optional
        if set_figsize = True, use this to manipulate figure size. The default is (10,5).
        
    left_zoom and right_zoom optionally allows you to cut-off a specified number of
    years at the beginning or end to 'zoom' in on specific years. Default is no zoom.

    """
    num_years = len(years)
    years = years[left_zoom:num_years-right_zoom]
    per_topic_vals = per_topic_vals[left_zoom:num_years-right_zoom]

    if set_figsize:
        plt.figure(figsize = figsize)
    #per_topic_vals = [per_year_size_dict[year][topic_id] for year in years]
    plt.scatter(years, per_topic_vals)
    plt.plot(years, per_topic_vals)
    plt.xticks(fontsize = 12, rotation = x_tick_rotation)
    plt.yticks(fontsize = 12)
    if plot_title:
        if title is None:
            title = "Topic %d" %topic_id
        plt.title(title, fontsize = 22, pad = 15)
    if max_y_val is not None:
        plt.ylim(0,max_y_val)
    else:
        plt.ylim(0,)
    if plot_ylabel:
        plt.ylabel("Topic Size", fontsize = 18)
    if plot_xlabel:
        plt.xlabel("Year", fontsize = 18)

        
        


## Plot subset or possibly all topics from a single model
def _topic_by_year_grid_plotter(per_year_size_dict, topics_to_plot_list, sizetype = None,
                                left_zoom = 0, right_zoom = 0, x_tick_rotation = 0,
                                save_fig = False, fig_outpath = None, 
                                fig_name = "topics_over_time_grid", dpi = 200,
                                fig_override = False):
    """
    plots a grid of size at most 5 x 5 showing topic trends over time
    uses output from get_per_group_topic_size but will only work
    as intended if this function has been applied to groups defined by 
    continuous years.
    
    Adjusts grid depending on the nubmer of topics it is asked to plot.
    For more than 5 topics, the number of columns in the grid is always 5 but the rows
    are adjusted to be only as many as needed.
    
    
   
    Parameters
    ----------
    per_year_size_dict: dictionary as output by get_per_group_topic_size where
         label_list given to that function is a list of years (int values)

    topics_to_plot_list : list of ints of length at most 25
        
    sizetype : str, optional, one of "word_count", "doc_count", "mean"
        if given, will include this in label on the y axis

    left_zoom and right_zoom optionally allows you to cut-off a specified number of
    years at the beginning or end to 'zoom' in on specific years. Default is no zoom.
    Same zoom is applied to each plot
    
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py


    Returns
    -------
    list of lists of length K containing of topic sizes over time for each topic

    """
    gvals = list(per_year_size_dict.keys())
    K = len(per_year_size_dict[gvals[0]]) #should be one size per topic in each element of out_dict
    num_plots = len(topics_to_plot_list)
    assert num_plots <= K, "Length of topics_to_plot_list should be less or equal to than %d" % K
    assert num_plots <= 25, "I can only plot at most 25 topics at a time"
    assert all([type(k) == int and k >= 0 and k < K for k in topics_to_plot_list]), "invalid topic index in topics_to_plot_list"
    
    #get values over years for each topic
    per_topic_vals = [[per_year_size_dict[year][k] for year in gvals] for k in range(K)]
    
    #get maximum value for setting plot limit
    max_val = np.max(per_topic_vals)

    #get years covered
    years = np.arange(gvals[0],gvals[len(gvals)-1]+1)

    #figure out grid size and make some size dependent adjustments
    nrows, ncols = Helpers.figure_out_grid_size(num_plots = num_plots, num_cols = 4, max_rows = 5)
    
    if num_plots >= 5:
        fontsize = 30
        titlesize = 40
    else:
        fontsize = 20
        titlesize = 25
         
    
    plt.rcParams.update({'font.family':'serif'})
    figsize = (ncols * 4, nrows * 4)
    
    if sizetype is not None:
        ylab = "Topic Size (normalized %ss)" % sizetype.replace("_"," ")
    else:
        ylab = "Topic Size (normalized)"
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False, figsize=figsize)
    fig.text(0.5, -.07,  "Year", ha='center', fontsize = fontsize)
    fig.text(-.07, 0.5, ylab, va='center', rotation='vertical', fontsize = fontsize)
    fig.tight_layout(h_pad=6, w_pad = 6)

    for i in range(ncols * nrows):
        ax = plt.subplot(nrows,ncols,i+1)
        if i < num_plots:
            ind = topics_to_plot_list[i]
            _topic_sizes_by_year_plotter(years, per_topic_vals[ind],
                                         topic_id = ind , max_y_val = max_val * 1.1,
                                         set_figsize=False, plot_xlabel = False, 
                                         plot_ylabel = False, left_zoom = left_zoom,
                                         right_zoom = right_zoom,
                                         x_tick_rotation = x_tick_rotation)
        else:
            ax.set_visible(False)

    if nrows == 5 or nrows == 4:
        top = .90
    elif nrows == 3:
        top = .85
    elif nrows == 2:
        top = .80
    elif nrows == 1:
        top = .65
    
    plt.suptitle("Topics Over Time", fontsize = titlesize)
    plt.subplots_adjust(top= top)     


    if save_fig:
        figure_saver(fig_name = fig_name, 
                 outpath = fig_outpath,
                 dpi = dpi,
                 fig_override = fig_override,
                 bbox_inches = "tight")




    
    
def plot_topic_sizes_by_year(theta_mat, year_labels, sizetype, doc_lengths = None,
                             plot_all_topics = True, custom_topic_list = None,
                             left_zoom = 0, right_zoom = 0, x_tick_rotation = 0,
                             save_fig = False, fig_name = None, fig_outpath = None, 
                             dpi = 200,
                             fig_override = False):
    """    
    Parameters
    ----------
    theta_mat : overall theta matrix as output by get_document_matrices
    
    year_labels : list or 1-D array of year labels for each document 
    
    sizetype : one of "word_counts","doc_counts","mean"
    
    doc_lengths : if word_counts specified, must also supply length of each 
          documentas obtained by _get_doc_lengths()
    
    plot_all_topics : Bool, optional
        plots all topics as given in theta matrix. The default is True.
    
    custom_topic_list : list of ints, optional
        if given and plot_all_topics is False,
        plots a custom list of topics. The default is None.
        f not given and plot_all_topics is False, raises an error
    
    left_zoom and right_zoom optionally allow you to cut-off a specified number of
    years at the beginning or end to 'zoom' in on specific years. Default is no zoom.
    Same zoom is applied to each plot
      
    x_tick_rotation : optionally rotate x tick marks
    
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

         if fig_name = None, figure naming options are handled automatically - each grid file is named
         K_topics_over_time_grid#
         
         else, fig_name is used and appended with 0, 1, 2... for the number of grids
   

    Returns
    -------
    None.

    """
    if not plot_all_topics:
        assert custom_topic_list is not None, "If plot_all_topics = False, custom_topic_list must not be None"
    
    per_year_size_dict = LdaOutput.get_per_group_topic_size(theta_mat = theta_mat, 
                                   label_list = year_labels,
                                   sizetype = sizetype, 
                                   doc_lengths = doc_lengths,
                                   normalized = True) #doesn't make sense to plot this without normalizing
    K = theta_mat.shape[0]
    
    #figure out how many grids to create    
    if plot_all_topics:
        topics_to_plot = list(range(theta_mat.shape[0]))
    else:
        topics_to_plot = custom_topic_list
    grids_list = Helpers.get_grids(topics_to_plot, num_row = 5, num_col = 4)

    

    #Call plot helper
    for i, plot_list in enumerate(grids_list):
        if fig_name is None:
            filename = "%d_topics_over_time_grid_%d" %(K, i)
        else:
            filename = fig_name + str(i)
        _topic_by_year_grid_plotter(per_year_size_dict, plot_list, 
                                    sizetype = sizetype,
                                    left_zoom = left_zoom, right_zoom = right_zoom,
                                    x_tick_rotation = x_tick_rotation, 
                                    save_fig = save_fig,
                                    fig_outpath = fig_outpath, 
                                    fig_name = filename,
                                    dpi = dpi,
                                    fig_override = fig_override)






### TOPIC SIZE OVER TIME + WORD BAR PLOT FOR ONE TOPIC


def plot_barplot_and_timeplot(model, per_year_size_dict, theta_mat, topic_id,
                              corpus, dictionary, figsize = (14,10),
                              topn = 20, lamb = 0.6, value_type = "counts",  #barplot parameters
                              plot_suptitle = True, custom_suptitle = None,
                              right_zoom = 0, left_zoom = 0, sizelabel = None,
                              detect_max_vals = True, #overtime plot parameters
                              save_fig = False, fig_outpath = None, 
                              fig_name = "bar_and_timeplot", dpi = 200,
                              fig_override = False): 
    
    """    
    Parameters
    ----------
    model : gensim LDA model
    
    per_year_size_dict: dictionary as output by get_per_group_topic_size where
         label_list given to that function is a list of years (int values)
    
    theta_mat : overall theta matrix as output by get_document_matrices
    
    corpus : gensim corpus
        corpus used to train LDA models
        (vector-count representation)
             
    dictionary : gensim dictinary 
        dictionary used to train LDA models
      
    figsize : tuple, optionally set figure size
    
    barplot parameters:
        topn : number of words to display
        lamb : lambda to use in relevance calculations
        value_type : whether to plot expected counts or fitted probabilites


    time plot parameters:             
        sizelabel :  optionally
            specify what kind of size calculation was used (for labelling purposes)
            not constrained to [word_count, doc_count, mean] because might also want to
            add in the word "normalized". Tile will be "<sizetype> over time" and 
            any _ will be replaced by " ". If not specified, over time plot has no title
            
     
        left_zoom and right_zoom optionally allow you to cut-off a specified number of
        years at the beginning or end to 'zoom' in on specific years. Default is no zoom.
        
        
    title parameters
        plot_suptitle : if False, plots no overall title
        custom_suptitle : Optionally, specify your own overall 
        
        
    detect_max_vals : bool
    
            if True, scales each plots relative to each topic in the model
            
            That is, looks at per_year_size_dict to find the maximum value that
            occurs in that dictionary and sets y axis of time plot accordingly.
            Similarly, looks at all magnitudes used in bar plots and sets
            axis accordingly.
            
            Do this to make plots for multiple topics easier to compare and
            to avoid being misled by axis differences
    
        
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

    
    Returns
    -------
    None.

    """
    plt.rcParams.update({'font.family':'serif'})
    
    
    if detect_max_vals:
        max_y_val = Helpers._get_max_size_in_dict(per_year_size_dict) *  1.1
    else: 
        max_y_val = None
    
    #a few initial checks
    K = model.num_topics
    assert theta_mat.shape[0] == K, "Model and theta_mat do not have same number of topics"
    assert topic_id < K, "topic_id must be less than K = %d" %K  
    
    years = list(per_year_size_dict.keys())
    assert len(per_year_size_dict[years[0]]) == K, "per_year_size_dict has different K from model"
    
    per_topic_vals = [per_year_size_dict[year][topic_id] for year in years]

    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(30,25)
    
    #over time plot
    fig.add_subplot(gs[3:22,13:30])
    
    #title of the over time plot
    if sizelabel is not None:
        plot_title = True
        title = sizelabel.replace("_"," ") + str(" by year")
    else:
        plot_title = False,
        title = ""
    
    _topic_sizes_by_year_plotter(years, 
                                 per_topic_vals, 
                                 topic_id,
                                 set_figsize = False, 
                                 plot_title = plot_title,
                                 title = title,
                                 left_zoom = left_zoom,
                                 right_zoom = right_zoom,
                                 max_y_val = max_y_val)   
    #barplot
    fig.add_subplot(gs[0:25,0:10])
    LdaOutputWordPlots.topic_relevance_barplot(model = model, 
                                               topicid = topic_id, 
                                               corpus = corpus, dictionary = dictionary,
                                               lamb = lamb,
                                               value_type = value_type, 
                                               theta_mat = theta_mat, 
                                               topn = topn, 
                                               detect_max_x = detect_max_vals,
                                               plot_title = True,
                                               title = "%d most relevant words ($\lambda$ = %s)" % (topn, str(lamb)),
                                               set_figsize = False, 
                                               save_fig = False,
                                               )
    #overall title:
    if plot_suptitle:
        if custom_suptitle is None:        
            plt.suptitle("Topic %d from %d-Topic Model" % (topic_id, K), fontsize = 30)
        else:
            plt.suptitle(custom_suptitle, fontsize = 30)
        plt.subplots_adjust(top= .85)    

    if save_fig:
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                         bbox_inches = "tight")
    





### COMPARING MULTIPLE MODELS TO ONE CENTRAL MODEL 



def _topic_time_comparison_plotter(main_dict, K_main,
                                   per_year_size_dicts, K_vals,
                                   topics_to_plot_list, 
                                   most_sim_topic_ids,
                                   most_sim_topic_vals,
                                   distance,
                                   left_zoom = 0, right_zoom = 0,
                                   save_fig = False, 
                                   fig_outpath = None, 
                                   fig_name = "topic_size_by_year_model_comparison", dpi = 200,
                                   fig_override = False):
    """
    ***Note: not recommended to compare more than 3 models at a time 
    - gets very crowded and legend may not align well***
    
    helper function for topic_size_by_year_model_comparison()
    which plots a single one of the grids plotted by that function
    
    see that function for more info 
    
    Parameters
    ----------
    main_dict : dictionary of topic sizes by year for the main model 
    
    K_main : number of topics for main model
    
    per_year_size_dicts : list of dictionaries of topic sizes by year for the
        models to be used in comparison
    
    K_vals : list of numbers of topics for each comparison model
    
    topics_to_plot_list : list of topic indices for topics to plot - can be at most 16
    
    most_sim_topic_ids : list of lists where each list contains the topic ids of the topics from the
        comparison model that are most similar to each topic in them ain model
    
    most_sim_topic_vals : lis tof lists corresponding to most_sim_topic_ids only containing the similarity
        metric values
    
    distance : str, one of: jensen_shannon (default) or kullback_leibler
            distance metric used to calculate similarity between topics
        
    left_zoom and right_zoom optionally allows you to cut-off a specified number of
        years at the beginning or end to 'zoom' in on specific years. Default is no zoom.
        Same zoom is applied to each plot
      
  

    Returns
    -------
    None.

    """
    assert distance in ["kullback_leibler", "jensen_shannon"]
    assert all([d.keys()== main_dict.keys() for d in per_year_size_dicts]), "dictionaries must all have same keys"
    
    #year values - assumes continuous!!!
    gvals = list(main_dict.keys())
    years = np.arange(gvals[0],gvals[len(gvals)-1]+1)
    num_years = len(years)
    assert all(gvals == years), "years from dictionaries must be continuous and in order"
    
    #number of plots to plot + figure out grid for it
    num_plots = len(topics_to_plot_list)
    assert num_plots <= 20, "I can only plot at most 20 plots"
    nrows, ncols = Helpers.figure_out_grid_size(num_plots, num_cols = 4, max_rows = 5) #max 20 x 16 grid
    
    #get values over years for all main model topics
    main_per_topic_vals = [[main_dict[year][k] for year in gvals] for k in range(K_main)]

    #get values over years for all the other models -- list contains entry for each model
    #and each entry contains an entry for each topic and
    #each topic entry is a list of topic sizes over years
    list_of_per_topic_vals = [[[d[year][k] for year in gvals] for k in range(K)] for K, d in zip(K_vals, per_year_size_dicts)]

    #get maximum value that occurs anywhere so plots all on same scale
    max_val = Helpers._get_max_size_in_dict_list(per_year_size_dicts + [main_dict])
    
    
    #overall plotting parameters
    if num_plots <= 4:
        if num_plots >= 3:
            fontsize = 20; top = 0.81; titlesize = 26; legend_loc = (.80,.92); legendsize = 16; xheight = -.06
        elif num_plots == 2:
            fontsize = 18; top = 0.68; titlesize = 26; legend_loc = (.80,.86); legendsize = 16; xheight = -.13
        elif num_plots == 1:
            fontsize = 16; top = 0.75; titlesize = 26; legendsize = 16; xheight = -.10
    else:
        if nrows == 2:
            fontsize = 35; top = 0.80; titlesize = 40; legend_loc = (.82,.89); legendsize = 25; xheight = -.06
        elif nrows == 3:
            fontsize = 35; top = 0.83; titlesize = 40; legend_loc = (.80,.91); legendsize = 30; xheight = -.06
        elif nrows == 4:
            fontsize = 35; top = 0.88; titlesize = 40; legend_loc = (.80,.93); legendsize = 30; xheight = -.03
        elif nrows == 5:
            fontsize = 35; top = 0.90; titlesize = 40; legend_loc = (.80,.94); legendsize = 30; xheight = -.03
            
            
    plt.rcParams.update({'font.family':'serif'})
    figsize = (ncols * 6, nrows * 7)

    #set up subplot grid with overall labels
    kvals_as_str = ", ".join([str(k) for k in K_vals])
    xlab = "%d-Topic Model" % K_main 
    xlab += " matched to most similar topics from (" + kvals_as_str + ") topic models"
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False, figsize=figsize)
    fig.text(0.5, xheight,  xlab, ha='center', fontsize = fontsize)
    fig.text(-.07, 0.4, "Topic Size (normalized)", va='center', rotation='vertical', fontsize = fontsize)
    fig.tight_layout(h_pad=14, w_pad = 6)
    years = years[left_zoom : num_years-right_zoom]
    
    #main plotting
    for i in range(ncols * nrows):
        ax = plt.subplot(nrows,ncols,i+1)
        if i < num_plots:
                #plot main topic
                main_vals = main_per_topic_vals[topics_to_plot_list[i]][left_zoom : num_years-right_zoom]
                plt.scatter(years, main_vals, 
                            alpha = 0.8, 
                            label = "%d-Topic Model" % K_main, 
                            s = 50)
                plt.plot(years, main_vals, alpha = 0.7)

                #plot most similar topics
                sim_inds = [] #store indices of most similar topics for labelling purposes
                dist_vals = [] #store KL values of most similar topics for labelling purposes
                for j, lst in enumerate(list_of_per_topic_vals): #for each comparison model
                    #access most similar topic ind
                    #j gets the list of most similar topics for that model
                    #then from there, we get the topic to plot
                    ind = most_sim_topic_ids[j][topics_to_plot_list[i]] 
                    dist = most_sim_topic_vals[j][topics_to_plot_list[i]] #for labelling
                    sim_inds.append(str(ind))
                    dist_vals.append(str(np.round(dist, 2)))
                    size_vals = lst[ind][left_zoom : num_years-right_zoom]   
                    plt.scatter(years, size_vals, 
                                alpha = 0.7, 
                                label = "%d-Topic Model" % K_vals[j], 
                                s = 50)
                    plt.plot(years, size_vals, 
                             alpha = 0.7)

                #title includes information about which topics matched and their KL or JS values
                if distance == "jensen_shannon":
                    combo_list = ["" + a + ": JS=" + b + ""  for (a,b) in  zip(sim_inds,dist_vals)]
                elif distance == "kullback_leibler":
                    combo_list = ["" + a + ": KL=" + b + ""  for (a,b) in  zip(sim_inds,dist_vals)]
                
                title = "Topic %d \n\n (" % topics_to_plot_list[i] + ", ".join(combo_list) + ")"
                plt.title(title, fontsize = 20, pad = 10)
                plt.ylim(0, max_val*1.1)
                plt.xticks(fontsize = 13)
                plt.yticks(fontsize = 13)
                #plt.xlabel("Year")
                handles, labels = ax.get_legend_handles_labels()
                
        else: #set remainder plots in grid to blank
                ax.set_visible(False)

                
    #overall legend
    if num_plots == 1:
        plt.legend(fontsize = legendsize)
    else:
        fig.legend(handles, labels, loc = legend_loc, fontsize = legendsize)

    #overall title
    suptitle = "Topics Over Time Model Comparison"
    plt.suptitle(suptitle, fontsize = titlesize)
    plt.subplots_adjust(top= top)     

    if save_fig:
        figure_saver(fig_name = fig_name, 
                 outpath = fig_outpath,
                 dpi = dpi,
                 fig_override = fig_override,
                 bbox_inches = "tight")




def topic_size_by_year_model_comparison(year_labels,
                                    main_model, theta_mat_main, model_list, theta_mat_list,
                                    sizetype, distance = "jensen_shannon", doc_lengths = None,
                                    plot_all_topics = True, custom_list = None,
                                    left_zoom = 0, right_zoom = 0,
                                    save_fig = False, 
                                    fig_name = None, 
                                    fig_outpath = None, 
                                    dpi = 200,
                                    fig_override = False):
    """
    This function plots topic size over years for the main model in the same way that 
    plot_topic_sizes_over_years() does but it also adds topic size over time for topics
    from each model in model_list. Because topic IDs across models don't per se
    correspond to the same topics, this function calculates the KL divergence between
    each topic in each model of model_list and each topic of main_model and picks for each
    topic of main_model the topics from the models in model_list that have lowest KL
    divergence from the main_model topic
    
    Note that in KL divergence calculations, q is always assigned to be the model with fewer
    topics. See get_most_sim_topics() function 
    

    Parameters
    ----------
    year_labels : list or array of same length as number of documents in corpus
        used to train models, containing year labels (as ints) for each document
        
    main_model : gensim LDA model
        the main model to consider - other models are compared to it
        
    theta_mat_main : the theta matrix as otuput by get_document_matrices 
        corresponding to the main model
 
    model_list : a list of gensim LDA models
        models to compare to main model
        
    theta_mat_list : list of numpy arrays
        theta matrices as output by get_document_matrices for each of the comparison models
            this should be in same order as model_list
            
    sizetype : str, one of: word_count, doc_count, mean
        topic size metric to use 
        
    distance : str, one of: jensen_shannon (default) or kullback_leibler
        distance metric used to calculate similarity between topics
        
    doc_lengths : if sizetype = word_count, must also specify length of each document in corpus
            default is None.
            
    plot_all_topics : bool, optional
        if true, plots 4x4 grids to cover all topics in main_model. The default is True.
    
    custom_list : list of ints, optional
        custom list of topic ids from main model to plot. The default is None.
    
    left_zoom and right_zoom optionally allows you to cut-off a specified number of
        years at the beginning or end to 'zoom' in on specific years. Default is no zoom.
        Same zoom is applied to each plot
      
    
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py
        
        if fig_name is none, figure name is generated automatically and takes form
        "K_size_by_year_comparison_i" where K is number of topics in
        main model and i is the ith grid created for that model

    Returns
    -------
    None.

    """
    assert distance in ["kullback_leibler", "jensen_shannon"]
    
    if not plot_all_topics:
        assert custom_list is not None, "if plot_all_topics = False, must specify custom_list of topics to plot"
        

    #check model_list and theta_mat_list appear to correspond
    K_main = main_model.num_topics
    K_vals1 = [m.num_topics for m in model_list]
    K_vals2 = [mat.shape[0] for mat in theta_mat_list]
    assert K_vals1 == K_vals2, "model_list and theta_mat_list do not correspond - K mismatch detected"
    
    #get all the dictionaries of topic sizes for each year
    per_year_size_dicts =  [LdaOutput.get_per_group_topic_size(matrix, 
                                       label_list = year_labels, 
                                       sizetype = sizetype, 
                                       doc_lengths = doc_lengths,
                                       normalized = True) for matrix in theta_mat_list]
    #get dictionary for the main function
    main_dict = LdaOutput.get_per_group_topic_size(theta_mat_main, 
                                       label_list = year_labels, 
                                       sizetype = sizetype, 
                                       doc_lengths = doc_lengths,
                                       normalized = True) 

    #main dict is the one the others get matched to
    #for each model in model_list, find tind the topics most similar to each of the main model topics
    # (there can be repeats). Each entry of list below is in order 0...K_main
    similarity_output = [LdaOutputTopicSimilarity.get_most_sim_topics(main_model, m, 
                                                                      distance = distance,
                                                                      return_vals = True) for m in model_list]
    most_sim_topic_ids = [elem[0] for elem in similarity_output]
    #the actual KL divergence values - lower is more similar
    most_sim_topic_vals = [elem[1] for elem in similarity_output] 
    
    #divide topics to plot list into lists of at most 16
    if plot_all_topics:
        topics_to_plot_list = list(range(K_main))
    else:
        topics_to_plot_list = custom_list
        
    grids_list = Helpers.get_grids(topics_to_plot_list, num_row = 5, num_col = 4)
    for i, grid in enumerate(grids_list):
        if fig_name is None:
            filename = "%d_size_by_year_comparison_%d" %(K_main, i)
        else:
            filename = fig_name + str(i)
        _topic_time_comparison_plotter(main_dict = main_dict, K_main = main_model.num_topics,
                                       per_year_size_dicts = per_year_size_dicts, K_vals = K_vals1,
                                       topics_to_plot_list = grid,
                                       most_sim_topic_ids = most_sim_topic_ids,
                                       most_sim_topic_vals = most_sim_topic_vals,
                                       distance = distance,
                                       left_zoom = left_zoom, right_zoom = right_zoom,
                                       save_fig = save_fig, fig_outpath = fig_outpath, 
                                       fig_name = filename,
                                       dpi = dpi,
                                       fig_override = fig_override)
    









###### WORDS WITHIN TOPIC OVER TIME



def plot_topic_by_year(per_year_topics_dict, topic_id, model, corpus, dictionary, 
                       lamb = 0.6, topn = 20, left_zoom = 0, right_zoom = 0,
                       color = "purple", alpha = 0.3,
                       figsize = (20,20),
                       save_fig = False, fig_outpath = None, 
                       fig_name = "topics_by_year", dpi = 200,
                       fig_override = False):
    """
    visualizes how words within a single topic change over years
    using output from get_per_group_topics()
    
    does this using a grid with:
        top topn most relevant words along y axis
        years along x axis
        circles with size proportional to probability of each word in each year
            

    Parameters
    ----------
    per_year_topics_dict : output from from get_per_group_topics()
 
    topic_id : positive int < model.num_topics
        indicates the topic to visualize
        
    model : LDA gensim model
    
    corpus : gensim corpus
        corpus used to train LDA models
        (vector-count representation)
             
    dictionary : gensim dictinary 
        dictionary used to train LDA models
        
    lamb : float, optional
        lambda value to use in relevance calculation. The default is 0.6.
        
    topn : int, optional
        plot will display the topn most relevant words. The default is 20.
        Warning: very large and very small values of topn may yield 
        less appealing plots
        
    left_zoom and right_zoom optionally allows you to cut-off a specified number of
        years at the beginning or end to 'zoom' in on specific years. Default is no zoom.
        
    figsize : tuple, optional
        set figure size
        
    color : color of circle markers, optional
        
    alpha : float, optional
        transparency of circle markers. The default is 0.3.
        

    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

    Returns
    -------
    None.

    """
    assert type(per_year_topics_dict) == dict, "per_year_topics_dict must be a dictionary as output by get_per_group_topics()"
    K = model.num_topics
    assert type(topic_id) == int and topic_id >= 0 and topic_id < K, "invalid topic ID"
    
    
    
    yvals = list(per_year_topics_dict.keys())
    #apply zooming if needed
    yvals = yvals[left_zoom:len(yvals)-right_zoom]
    
    #get year-specific topic distribution for topic <topic_id> for each desired year
    topic_by_year = np.array([per_year_topics_dict[year][topic_id] for year in yvals])
    
    
    #get top most relevant words for topic overall
    rel_mat, order_mat = LdaOutput.get_relevance_matrix(model = model, 
                                                        corpus = corpus, 
                                                        dictionary = dictionary, 
                                                        lamb = lamb,
                                                        phi = None)
    top_inds = order_mat[topic_id][:topn] #top words for that specific topic
    top_overall_word_list = [dictionary[i] for i in top_inds] #get corresponding words
    
    
    #get probabilities for top words for every year
    #will still line up even with zooming because topic_by_year only has len(yvals) elements
    top_overall_word_probs_by_year = [np.array([topic_by_year[year_ind][word_ind] for word_ind in top_inds]) for year_ind in range(len(yvals))]
    
    
    #plot scatterplot grid with word probabilities determining size of dot
    plt.rcParams.update({'font.family':'serif'})
    plt.figure(figsize = figsize)
    plt.grid(b = True, axis = "both", alpha = 0.30) #add grid lines to make it easier to see
    vert_range = np.linspace(0,1,num = topn)
    for i, year in enumerate(yvals):
        plt.scatter(np.repeat(year,topn), vert_range, color = color, 
                   alpha = alpha, marker = "o", edgecolor = 'k', linewidth = 1,
                   s = np.flip(np.round(10000 * top_overall_word_probs_by_year[i],0) )) 
        
        #Note on 10000 here:
        #means that e.g. .0500 --> 500 and .1000 -> 1000
        #so if word A has probability .05 and word B has probability .1, area of
        #circle is, as expected, twice as large in area (since that is how s is defined)
        #limitation of this plot is that circle area is hard to judge visually
        #but circles can still give general idea of words that are large/small
        #in probability
        #rounding means that e.g. .0500 and .05005 are indistinguishable

    plt.yticks(ticks = vert_range, labels = np.flip(top_overall_word_list), 
               rotation = 0, 
               fontsize = 23)
    plt.xticks(fontsize = 22)
    plt.xlabel("Year",fontsize = 30)
    plt.title("Topic %d Over Time: Expected counts by year \n for top %d overall most relevant words in topic" % (topic_id, topn),
              fontsize = 30,
              pad = 30)
    
    if save_fig:
        figure_saver(fig_name = fig_name, 
                 outpath = fig_outpath,
                 dpi = dpi,
                 fig_override = fig_override,
                 bbox_inches = "tight")

    return(top_overall_word_probs_by_year)
    






