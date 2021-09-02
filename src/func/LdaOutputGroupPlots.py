# -*- coding: utf-8 -*-
"""

This file contains functions for processing the output of gensim LdaModels

Specifically, it contains functions for visualizing comparisons between groups of
documents (eg. topic size by group)

It draws on calculations implemented in LdaOutput.py 

Author: Kyla Chasalow
Last edited: August 19, 2021


"""
import pandas as pd
import numpy as np
import numpy_indexed as npi
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec


import Helpers
from Helpers import figure_saver
import LdaOutput
import LdaOutputWordPlots






#### GENERAL TOPIC BY GROUP FUNCTIONS
    
def plot_topic_size_by_group(per_group_size_dict, plot_type, group_name = None, xlab = None,
                            plot_topic_colors = False, legend_outside_plot = True,
                            save_fig = False, fig_outpath = None, 
                           fig_name = "topic_size_by_group", dpi = 200,
                           fig_override = False):
    """
    Contains three options for examining the relationship between group and topic size
    
    options for paramter plot_type:
    
        barplot_by_topic plots a barplot where each cluster of bars represents a topic
            and the bars within a cluster represent the groups. Easy to see if, for example,
            topic 10 is dominated by group B while topic 11 is dominated by group A
            
        barplot_by_group plots a barplot where each cluster of bars represents a group
            and the bars within a cluster represent the topics. Topics are only assigned
            different colors if plot_topic_colors = True and for large numbers of topics,
            this can get uninterpretable. If plot_topic_colors = False, all bars are blue
            and though it is harder to identify a particular topic (they are in order from 0
            to K so it is possible to count up from the left), the plot gives an overall sense 
            the distribution over topics for each group and whether it is different
            
        scatterplot_grid plots a pairplot comparing size of each topic between groups.
            points closer to the x=y line indicate similar sizes for those topics. Points
            off the x=y line indicate that topic is larger for one group and smaller for the other
            
        This plot is particularly useful as K gets large, for then the bar plots can get
            too crowded to be very useful.

    Parameters
    ----------
    per_group_size_dict : dictionary as output by get_per_group_topic_size()
        
    plot_type : str, either "barplot_by_topics", "barplot_by_group" or "scatterplot_grid"
        
    group_name : str, optional
        for bar plots, this gets added in title or as title of legend.
        The default is None and in that case, title/legend simply read "Group"
      
    xlab : str, optional
        optionally specify x label if using one of the bar plots (else this does nothing)
        for barplot_by_topic, default label if xlab is none is "Topic"
        for barplot_by_group, default is no label
      
    plot_topic_colors : bool, optional
        see description of barplot_by_group above. The default is False.
    
    legend_outside_plot : bool, optional
        if True, plots legend outside plot. Otherwise, plots legend within plot
        (using location = best functionality)
    
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

    Returns
    -------
    None.

    """
    
    types_list = ["barplot_by_topic", "barplot_by_group", "scatterplot_grid"]
    assert plot_type in types_list, "plot_type must be one of" + str(types_list)

    #common settings
    plt.rcParams.update({'font.family':'serif'})
    gvals = list(per_group_size_dict.keys())
    K = len(per_group_size_dict[gvals[0]])    
    if group_name is None:
        group_name  = "Group"

    #two that rely on df with columns = groups
    if plot_type == "barplot_by_topic" or plot_type == "scatterplot_grid":
        df = pd.DataFrame(per_group_size_dict)
        topic = list(range(df.shape[0]))
        df = pd.concat([df,pd.DataFrame(topic)], axis = 1)
        df.rename({0:"Topic"}, inplace = True, axis = 1)
        
        if plot_type == "barplot_by_topic":
            df.plot(x = "Topic",
                   y = df.keys()[0:len(gvals)],
                   kind = "bar",
                   figsize = (20,8),
                   alpha = .6,
                   color = None, # ["blue","orange","brown", "lightgreen"],
                   edgecolor = "black")
            #further plot aesthetics
            if xlab is None:
                xlab = "Topic"
            plt.xlabel(xlab, fontsize = 18)
            plt.ylabel("Normalized Measure of Topic Size", fontsize = 18)
            plt.xticks(rotation = 0, fontsize = 15)
            plt.title("Topic Size by %s for %d-Topic Model" %(group_name, K), pad = 15, fontsize = 25)
            plt.yticks(fontsize = 14)
            if legend_outside_plot:
                plt.legend(bbox_to_anchor=(1,1), loc="upper left",
                           fontsize = 18,
                           title = group_name, title_fontsize = 18)
            else:
                plt.legend(fontsize = 18, title = group_name, title_fontsize = 18)
            if save_fig:
                figure_saver(fig_name = fig_name, 
                             outpath = fig_outpath,
                             dpi = dpi,
                             fig_override = fig_override,
                             bbox_inches = "tight")
        #-------------------------------------------------------------------------------------------
        elif plot_type == "scatterplot_grid":
            
            g = sns.pairplot(df.iloc[:,0:len(gvals)], corner = True, 
                             plot_kws={"color":"purple",'alpha':.4, "s": 75, "edgecolor": "black"},
                             diag_kws = {"color":"purple", "alpha": .4})
            sns.set_context("notebook", rc={"axes.labelsize":18},font_scale= 0.9)
            g.fig.suptitle("Comparing Topic Sizes by Document Group for %d-Topic Model" % K,
                             y = 1.04, fontsize = 18)
           
            #set axis limits to all be the same - make it easier to compare to x=y line
            max_val = df.iloc[:,:len(gvals)].max().max() * 1.1
            for ax in g.axes.flatten():
                if ax is not None:
                    ax.set_ylim(bottom = 0, top = max_val)
                    ax.set_xlim(left = 0, right = max_val)
                    ax.tick_params(rotation = 35) #improve visibility
                   
        
            if save_fig:
                figure_saver(fig_name = fig_name, 
                             outpath = fig_outpath,
                             dpi = dpi,
                             fig_override = fig_override,
                             bbox_inches = "tight")
            return(g)
        
        
    #---------------------------------------------------------------------------------------------
    #relies on df with columns = topics
    elif plot_type == "barplot_by_group":
        list_by_topic = [[per_group_size_dict[g][k] for g in gvals] for k in range(K)]

        by_topic_dict = {}
        for k in range(K):
            by_topic_dict[str(k)] = list_by_topic[k] #str trick prevents 0th topic from being renamed "Group" below

        df = pd.DataFrame(by_topic_dict)
        df = pd.concat([df,pd.DataFrame(gvals)], axis = 1)
        df.rename({0:"Group"}, inplace = True, axis = 1)

        #can get overwhelming if have a lot of topics so give option to just plot all blue
        if plot_topic_colors:
            color = plt.get_cmap("tab20")(range(K))
            alpha = 0.6
            if K > 20:
                print("warning: plot_topic_colors = True not recommended for K > 20")
        else:
            color = "blue"
            alpha = 0.4
        
        df.plot(x = "Group",
               y = df.keys()[0:K],
               kind = "bar",
               figsize = (20,8),
               alpha = alpha,
               color = color,
               edgecolor = "black",
               legend = False)
    
        if xlab is None:
            xlab = " "
        plt.xlabel(xlab, fontsize = 18)
        plt.ylabel("Normalized Measure of Topic Size", fontsize = 18)
        plt.xticks(rotation = 0, fontsize = 17)
        plt.title("Topic Size by %s for %d Topic Model" %(group_name, K), pad = 15, fontsize = 25)
        plt.yticks(fontsize = 14)
        if plot_topic_colors:
            if legend_outside_plot:
                plt.legend(bbox_to_anchor=(1,1), 
                           loc="upper left",
                           fontsize = 14,
                           title = "Topic", title_fontsize = 14)
            else:
                plt.legend(fontsize = 15, title = "Topic", title_fontsize = 14)

        if save_fig:
                    figure_saver(fig_name = fig_name, 
                                 outpath = fig_outpath,
                                 dpi = dpi,
                                 fig_override = fig_override,
                                 bbox_inches = "tight")

   
    
    

#similar to plot_topic_size_over_years but that function is
#labeled/set-up specifically for years. This one is more general - any kind of
#grouping. This function would also be appropriate for non-continuous years
#something plot_topic_size_over_years can't handle


def plot_topic_by_group(per_group_topics_dict, topic_id, model, corpus, dictionary, 
                       lamb = 0.6, topn = 20, magnifier = 10000, color = "purple", alpha = 0.4,
                       color_by_group = False, group_label = None, group_label_list = None,
                       figsize = None, xtick_rotation = 0,
                       save_fig = False, fig_outpath = None, 
                       fig_name = "topics_by_group", dpi = 200,
                       fig_override = False):
    """
    visualizes how words within a single topic change across groups
    using output from get_per_group_topics()
    
    does this using a grid with:
        top topn most relevant words along y axis
        group labels along x axis
        circles with size proportional to probability of each word in each group
            
    WARNING: This function plots the overall most relevant words for the topics
    and compares the group using those words. But the overall most relevant words
    might not correspond to the most relevant words WITHIN a particular group

    Parameters
    ----------
    per_group_topics_dict : output from get_per_group_topics() 
 
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
        
    magnifier: how much to multiply each probability by when determining size of circles
        
    color : color of circle markers, optional
        
    alpha : float, optional
        transparency of circle markers. The default is 0.3.

    color_by_group : bool, optional
        if True, overrides color and alpha arguments and instead automatically
        assigns each group (each column of plot) a color. Not necessary to 
        distinguish groups since they are marked on x axis, but may be
        visually clearer. The default is False.
   
   xtick_rotation : optionally rotate group labels on x axis  
   
   group_label : str, optional
       optionally, specify name of group for use in title and xlabel
       if None, will just use the word "Group"
       
   group_label_list : list of strings
       if None, just uses group labels from per_group_topic_dict
       if specified, uses these labels instead
       
   figsize : tuple, optional
       optionally override auotmatic figsize mechanism to set own
        
   save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

    Returns
    -------
    None.

    """

    gvals = list(per_group_topics_dict.keys())
    G = len(gvals)
    if group_label_list is not None:
        assert len(group_label_list) == G, "group_label_list must contain %d values" % G 
    
    #get topic distribution for given topic for each group
    topic_by_group = np.array([per_group_topics_dict[group][topic_id] for group in gvals])

    #get top most relevant words for topic overall
    rel_mat, order_mat = LdaOutput.get_relevance_matrix(model = model, 
                                                            corpus = corpus, 
                                                            dictionary = dictionary, 
                                                            lamb = lamb,
                                                            phi = None)
    top_inds = order_mat[topic_id][:topn]
    top_overall_word_list = [dictionary[i] for i in top_inds]

    #get probabilities for top words for every group
    top_overall_word_probs_by_group = [np.array([topic_by_group[group_ind][word_ind] for word_ind in top_inds]) for group_ind in range(G)]

    #navigate some plotting options
    plt.rcParams.update({'font.family':'serif'})
    if figsize is None:
        height = np.min([topn,20])
        width = np.min([np.round(G*2.5),20])
        figsize = (width,height)
        
    plt.figure(figsize = figsize)
    plt.grid(b = True, axis = "both", alpha = 0.30, zorder = 0) #add grid lines to make it easier to see
  
    #if true, color each group differently using automatic color funtionality when color = None
    if color_by_group:
        color = None
        alpha = 0.7
        
    if group_label is None:
        group_label = "Group"
    
    #set-up grid
    vert_range = np.linspace(0,1,num = topn)
    hori_range = np.linspace(0,1,num = len(gvals))


    #main plotting
    for i, group in enumerate(gvals):
        plt.scatter(np.repeat(hori_range[i], topn), 
                    vert_range,
                    zorder = 2,
                    alpha = alpha, 
                    color = color,
                    marker = "o", 
                    edgecolor = 'k', 
                    linewidth = 1,
                    s = np.flip(np.round(magnifier * top_overall_word_probs_by_group[i],0) )
                     ) 


    #labels
    plt.yticks(ticks = vert_range, labels = np.flip(top_overall_word_list), 
                   rotation = 0, 
                   fontsize = 16)
    if group_label_list is None:
        group_label_list = gvals
    plt.xticks(ticks = hori_range, 
               labels = group_label_list,
               fontsize = 20,
               rotation = xtick_rotation)
    plt.xlim(-.4,1.4) #add a little buffer to edges
    plt.xlabel(group_label,
               fontsize = 25)
    plt.title("Topic %d by %s: \n \n Normalized expected counts for top %d overall most relevant words in topic" % ( topic_id, group_label, topn),
              fontsize = 25,
              pad = 30)


    if save_fig:
                figure_saver(fig_name = fig_name, 
                             outpath = fig_outpath,
                             dpi = dpi,
                             fig_override = fig_override,
                             bbox_inches = "tight")





#FUNCTION TO INSTEAD ISOLATE TOP WORDS BY GROUP 
#the above looks only at overall most relevant words
def topic_words_by_group_grid_plotter(topic_by_group_dict, value_type, title, ylabel, 
                                      custom_groups =  None,
                                      plot_overall_magnitudes = True, plot_title_and_legend = True,
                                      save_fig = False, fig_outpath = None, 
                                      fig_name = None, dpi = 200,
                                      fig_override = False):
    """
    Helper function to plot a grid(s) representing single topic split by group 
    Grids contain at most 6 groups so if number of groups > 6, get multiple grids
    
    Parameters
    ----------
    topic_by_group_dict : a dictionary for a single topic as output by 
        LdaOutput.get_group_topword_vals_for_topic()

    value_type : str, one of "counts" or "probabilities"
        in this case, this doesn't affect the values represented, which are 
        in topic_by_group_dict already, but just tells function what kind of values
        are in topic_by_group_dict and affects coloring of barplot
    
    custom_groups : a list containing a subset of keys in topic_by_group_dict
        optionally tell plot to only plot some of the groups containing in 
        topic_by_group_dict (e.g. if had data over many years and wanted
        topic for every decade, could specify custom_groups = [range(1970,2021,10)]
                             
    title : str, title of plot

    ylabel : str, y axis label, x axis is set automatically according to
        value_type

    plot_overall_magnitudes : bool, optional
        optionally turn off the plotting of the overall group-corpus counts/probabilities
        default is True.

    plot_title_and_legend : bool, optional
        optionally turn off title and legend. The default is True.
        
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py
        
        Note: automatically adds a number to the end of fig_name 
        to mark the grid. If there are fewer than 6 groups, this is a bit
        redundant, but for > 6, function creates multiple grids of 6 and this
        ensures they don't override each other
   
    Returns
    -------
    None.

    """   
    #name settings
    if fig_name is None:
        fig_name = "topic_by_group"
        
    #get group info
    groups = list(topic_by_group_dict.keys())
    #if custom groups specified, check that every group there is in groups
    if custom_groups is not None:
        assert np.all([elem in groups for elem in custom_groups]), "custom group labels must all be in topic_by_group_dict() keys"
        groups = custom_groups # plot only the custom groups going forward
    num_groups = len(groups)
 
    
    #split into grids of at most 6 groups at a time
    grids = Helpers.get_grids(to_plot_list = list(range(num_groups)),
                      num_col = 2, num_row = 3)
   
    #plot grid for each set of 6 groups
    for i, grid in enumerate(grids):
                
        #get subset of groups to put in this grid
        group_subset = [groups[elem] for elem in grid]
        
        #extract values in form needed by plotter
        word_lists = [topic_by_group_dict[g][0] for g in group_subset]
        topic_magnitude_lists = [topic_by_group_dict[g][1] for g in group_subset]
        overall_magnitude_lists = [topic_by_group_dict[g][2] for g in group_subset]
        
        LdaOutputWordPlots.topic_grid_plotter(word_lists = word_lists,
                                      topic_magnitude_lists = topic_magnitude_lists,
                                      to_plot = list(range(len(group_subset))),
                                      label_list = group_subset,
                                      value_type = value_type,
                                      overall_magnitude_lists = overall_magnitude_lists,
                                      plot_overall_magnitudes = plot_overall_magnitudes,
                                      title = title, 
                                      ylabel = ylabel,
                                      plot_title_and_legend = plot_title_and_legend)
        #saving options
        if save_fig:
                   figure_saver(fig_name = fig_name + "_%d"%i, 
                                outpath = fig_outpath,
                                dpi = dpi,
                                fig_override = fig_override,
                                bbox_inches = "tight")
       
        

    
def plot_topic_words_by_group(wordtopic_array, group_list, to_plot_list, 
                                  value_type, corpus, dictionary, custom_groups = None,
                                  topn = 20, lamb = 0.6, group_name = None, 
                                  plot_title_and_legend = True, plot_overall_magnitudes = True,
                                  save_fig = False, fig_outpath = None, 
                                  fig_name = None, dpi = 200,
                                  fig_override = False):
    """
    
  
    1. Input document-specific matrices of expected word counts as obtained
    from LdaOutput.get_document_matrices (the wordtopic arrays) along
    with group labels for each document and some information from corpus and dictionary 
    
    2. Group these documents using labels in group_list via  LdaOutput.get_per_group_topics()
    to get group-specific K x V topic matrices either in form of expected counts for each (k,v)
    combo or normalized to get probabilities
    
    3. For every topic in to_plot_list, plot a grid of barplots with one per group in group_label
    

    Parameters
    ----------
    wordtopic_array : (D, K, V) shaped array as output by get_document_matrices with
        per_word_topics set to True
 
    group_list : length D list of group labels for each document

    to_plot_list : list of integers OR string "all" if would like to create grids for
        all topics

    custom_groups : a list containing a subset of keys in topic_by_group_dict
        optionally tell plot to only plot some of the groups containing in 
        topic_by_group_dict (e.g. if had data over many years and wanted
        topic for every decade, could specify custom_groups = [range(1970,2021,10)]

    value_type : str, one of "counts" or "probabilities"
        if counts, barplots are in red and blue and represent expected counts
        if probabilities, barplots are in purple and green and represent probabilities

    corpus: gensim corpus 
        corpus used to train LDA model that generated wordtopic_array
        (vector-count representation)
        
    dictionary : gensim dictinary 
        dictionary used to train LDA model that generated wordtopic_array
   
    topn : int, optional
        the number of words to show in each barplot. The default is 20.
        
    lamb : float or int in [0,1], optional
        parameter which determines relevance calculation. The default is 0.6.
        see LdaOutput.get_relevance_matrix() for more
        
    group_name : a string to be used in labeling the title and y axis
        as in "Topics by <group_name>". For example, might give it "Journal"
        if labels are different journals
                
    plot_title_and_legend : bool, optional
        optionally turn off title and legend. The default is True.
        
    plot_overall_magnitudes : bool, optional
        optionally turn off the overall corpus counts/probabilities (in blue/green)
        to only plot the topic-specific ones (in red/purple). The default is True.
        
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py
        
        if fig_name is None, uses title "K_topword_by_group_grid_i" where K is
            number of topics in model and i is the topic_id for that plot
            
        Note that each topic in to_plot_list is saved as a separate figure
        either with this naming convention or with the given fig_name with i 
        appended to the name


    Returns
    -------
    None.

    """
    
    #checks and initial set-up
    K = wordtopic_array.shape[1]
    options = ["counts", "probabilities"]
    assert value_type in options, "value_type must be one of " + str(options)
    if type(to_plot_list) == str:
        assert to_plot_list == "all", "to_plot_list must be either a list of ints or 'all'"
        to_plot_list = list(range(K))
    if value_type == "counts":
        normalized = False
    else:
        normalized = True
    if group_name is None:
        group_name = "Group"
    if fig_name is None:
       fig_name = "%d_topword_by_group_grid_" % K

    #extract topics per group dict
    per_group_topics_dict = LdaOutput.get_per_group_topics(wordtopic_array, 
                                                      label_list = group_list,
                                                      normalized = normalized)
    
    #extract relevance matrix and order for each group
    relevance_dict = LdaOutput.get_group_relevance_dict(corpus = corpus,
                                   dictionary = dictionary,
                                   group_list = group_list,
                                   per_group_topic_dict= per_group_topics_dict,
                                   lamb = lamb)
    
    #extract top words and prob/counts for each group
    topword_dict = LdaOutput.get_topword_values_dict(corpus = corpus,
                                     dictionary = dictionary, 
                                     group_list = group_list, 
                                     per_group_relevance_dict = relevance_dict,
                                     per_group_topic_dict = per_group_topics_dict,
                                     normalized = normalized,
                                     topn = topn)

    #plot a grid for each topic in to_plot_list
    for i, topic_id in enumerate(to_plot_list):
        d = LdaOutput.get_group_topword_vals_for_topic(topword_dict, topic_id)
        topic_words_by_group_grid_plotter(d, 
                           value_type = value_type, 
                           custom_groups = custom_groups,
                           title = "Topic %d by %s" % (topic_id, group_name), 
                           ylabel = "20 most relevant words within %s ($\lambda =$ %s)" % (group_name.lower(), str(lamb)),
                           plot_title_and_legend = plot_title_and_legend,
                           plot_overall_magnitudes = plot_overall_magnitudes,
                           save_fig = save_fig,
                           fig_name = fig_name + str(i),
                           fig_override = fig_override,
                           dpi = dpi,
                           fig_outpath = fig_outpath)
    
    
















#--------------------------------------------
### GROUPS OVER TIME
#---------------------------------------

def _group_over_time_plotter(group_dict, year_dict,  zoom_dict = None,
                           figsize = (10,6), xlabel = None, ylabel = None, 
                           title = None, title_group_label = None,
                           legend_outside_plot = False, plot_legend = True, legend_label = None, legend_loc = (1,1),
                           set_figsize = True, set_ylabel = True, set_xlabel = True, set_title = True,
                           max_y_val = None):
    """
    Helper Function for plotting multiple groups' trajectories over time
    
    Parameters
    ----------
    group_dict : dictionary where keys are group names and values are arrays or lists containing the
        the topic over time values for each group (in order of years in year_dict for that group).
        
    year_dict : dictionary where keys are group names and values are arrays or lists containing the 
        possiblly different years for each group. The lengths of these arrays/lists should match
        the length of the corresponding entries in group dict

    zoom_dict : optional dictionary for zooming in on parts of the graph
        keys are group names and values are length 2 tuples of integers indicating
        the number of integers to cut off on the left and the number to cut off on the right
        
        For example:
            {"Journal of Magic":(0,3),
             "Journal of Unicorn Medicine":(1,0)}
        
        would indicate that for the Journal of Magic plot, the 3 most recent years
        (on the right) should not be plotted, while for Journal of Unicorn Medicine,
        the first year on the left should not be plotted
        
        this might be useful, for example, if data for year 2021 were only partial
        so didn't want to plot them...since groups may cover different year ranges,
        zoom_dict is used to allow flexibility
        
    
    set_figsize, set_ylabel, set_xlabel, set_title, plot_legend all turn off/on the 
        corresponding component of the graph
        
    figsize, ylabel, xlabel, and title are the corresponding values that are used
        if the above parameters are set to true. Default xlabel is "Year"
        and default ylabel is "Normalized topic size". Default title is 
        "Topic Size Over Time by <title_group_label>" 

    legend:  if legend_outside_plot is True, then legend is plotted outside
        of the plot (otherwise inside). If legend_label is specified, this will be used
        as the title of the legend. If legend_outside_plot = True, then can also specify
        legend_loc (a tuple with two values) to adjust where it is. Default is upper right corner (1,1)

    title_group_label : str
        used in the title,  "Topic Size Over Time by <title_group_label>" 
        if not specified, default is "Group"

    max_y_val : int or float, optional
        sets maximum value of y axis. The default is None.

    Returns
    -------
    None.

    """
    assert year_dict.keys() == group_dict.keys(), "key mismatch between year_dict and group_dict"
    if zoom_dict is not None:
        assert zoom_dict.keys() == year_dict.keys(), "key mismatch between zoom_dict and year_dict"
    
    #some overall set-up
    if set_figsize:
        plt.figure(figsize = figsize)
    plt.rcParams.update({'font.family':'serif'})
    
    #get list of groups
    gvals = list(group_dict.keys())
    
    #plot each group
    if zoom_dict is None:
        for g in gvals:
            plt.scatter(year_dict[g], group_dict[g], label = g)
            plt.plot(year_dict[g], group_dict[g])
    #handle zooming if needed        
    else:
        for g in gvals:
            num_years = len(year_dict[g])
            x = year_dict[g][zoom_dict[g][0]:num_years-zoom_dict[g][1]]
            y = group_dict[g][zoom_dict[g][0]:num_years-zoom_dict[g][1]]
            plt.scatter(x,y, label =g)
            plt.plot(x,y)
    
    #labeling
    if title_group_label is None:
        title_group_label = "Group"
        
    if plot_legend:   
        if legend_outside_plot:
            plt.legend(bbox_to_anchor=legend_loc, 
                               loc="upper left",
                               fontsize = 14,
                               title = legend_label, 
                               title_fontsize = 14)
        else:
            plt.legend(fontsize = 15, title = legend_label, title_fontsize = 14)

    if set_ylabel:
        if ylabel is None:
            ylabel = "Normalized Topic Size"
        plt.ylabel(ylabel, fontsize = 20)
    if set_xlabel:
        plt.xlabel("Year", fontsize = 20)
    if set_title:
        if title is None:
            title = "Topic Size Over Time by %s" % title_group_label
        plt.title(title, fontsize = 25, pad = 15)
    if max_y_val is not None:
        plt.ylim(0,max_y_val)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)





def _get_topic_by_group_time_values(theta_mat, group_list, year_list, doc_lengths, sizetype):
    """
    Get a dictionary of topic size over year values for each journal. Also get a 
    dictionary with a vector of years for each journal

    Parameters
    ----------
    theta_mat : np.array as output by get_document_matrices in LdaOutput.py
        contains LDA theta values for each document-topic combo (matrix is K x D)
    
    group_list : list of group labels for each document 

    year_list : list of year labels for each document
        
    doc_lengths : if sizetype is "word_count", then this must be specified: a list of
        the number of words in each document as output by LdaOutput.get_doc_lengths()
      
    sizetype : str
        one of "word_count", "doc_count", "mean"
        specifies what measure of topic size to use

    Returns
    -------
    0. dictionary:
          keys are group names
          each value is itself a dictionary with years as keys and topic size
               for topics 0...K (in order) for each year for that group
               (where K = theta_mat.shape[0])
     
    1. dictionary
          keys are group names
          values are the years (least to greatest) for which each group
          has observations
          
    2. np.array containing the unique group labels that occur in group_list
        equivalent to the keys of the two dictionaries described above

    """
    assert theta_mat.shape[1] == len(group_list), "group_list and theta_mat dimension mismatch"
    assert theta_mat.shape[1] == len(year_list), "year_list and theta_mat dimension mismatch"
   
    #Note: code below does rely on order of documents being the same in every list. That is, 
    #the columns of theta_mat, and the entries of group_list, the entries of year_list
    #must all correspond
    #It also relies on ________ outputing... in order
   
    #get the groups present in group_list
    groups = np.unique(group_list)
    
    #get document lengths grouped by journal - in dictionary form to make sure all matches up
    if doc_lengths is not None:
        labels, length_groups = npi.group_by(keys = group_list, values =  doc_lengths)
        doc_length_dict = {group:length for (group, length) in zip(labels, length_groups)}

    #get years grouped by journal - in dictionary form to make sure all matches up
    labels, year_groups = npi.group_by(keys = group_list, values =  year_list)
    year_dict = {group:years for (group,years) in zip(labels, year_groups)}

    #get theta matrix grouped by journal by first grouping indices 0...D
    theta_mat_dict = LdaOutput.get_theta_mat_by_group(theta_mat, group_list)
    
    #Apply function from LdaOutput to get topic size by year within each group
    per_group_year_dicts = {}
    for i, g in enumerate(groups):
        if doc_lengths is None:
            lengths = None
        else:
            lengths = doc_length_dict[g]
        per_group_year_dicts[g] =  LdaOutput.get_per_group_topic_size(theta_mat_dict[g], 
                                       label_list = year_dict[g],
                                       sizetype = sizetype, 
                                       doc_lengths = lengths,
                                       normalized = True)
        
    #get the years for which each journal has observations -- allowed to be different
    #note that np.unique() automatically sorts the years from earliest to latest
    unique_year_dict = {}
    for g in groups:
        unique_year_dict[g] = np.unique(list(per_group_year_dicts[g].keys()))
     
    return(per_group_year_dicts, unique_year_dict, groups)





def _get_topic_dict(topic_id, groups, per_group_year_dicts, unique_year_dict):
    """
    Helper function to  get the topic size for each group + year combo for a single topic
    given output of _get_topic_by_group_time_values()

    Parameters
    ----------
    topic_id : int, ID of topic to extract values for
    groups, per_group_year_dicts, and unique_year_dict are as output by  
    _get_topic_by_group_time_values

    Returns
    -------
    dictionary
        keys are groups
        values are the topic sizes by year (from earliest to latest) for 
        topic <topic_id> for each of the groups

    """
    per_topic_vals_dict = {}
    #Note: as long as unique_year_dict entries are in earliest to latest order (which they are)
    #these will be in earliest to latest year order
    for g in groups:
        per_topic_vals_dict[g] = [per_group_year_dicts[g][year][topic_id] for year in unique_year_dict[g]]
    return(per_topic_vals_dict)





def topic_by_group_time_plot(theta_mat, group_list, year_list, doc_lengths, sizetype, to_plot_list,
                        zoom_dict = None, detect_max_val = True,
                        set_figsize = True, figsize = (15,5),
                        legend_label = None, legend_outside_plot = False, legend_loc = (1,1), 
                        title_group_label = None, custom_title = None, 
                        save_fig = False, fig_outpath = None, 
                        fig_name = "topic_group_time_plot", dpi = 200,
                        fig_override = False):
    """
    THE main plotting function for plotting topics over time by group.
    If to_plot_list contains a single topic_id, 
    will plot for a single topic. If multiple given, will create grids of up to 
    16 topics at a time

    Parameters
    ----------
    theta_mat : np.array as output by get_document_matrices in LdaOutput.py
        contains LDA theta values for each document-topic combo (matrix is K x D)
    
    group_list : list of group labels for each document 

    year_list : list of year labels for each document
        
    doc_lengths : if sizetype is "word_count", then this must be specified: a list of
        the number of words in each document as output by LdaOutput.get_doc_lengths()
      
    sizetype : str
        one of "word_count", "doc_count", "mean"
        specifies what measure of topic size to use
  
    to_plot_list : list of ints
       contains topic IDs to plot
       
       shortcut: optionally give this argument the string "all" to
       plot all topics

    zoom_dict : optional dictionary for zooming in on parts of the graph
        keys are group names and values are length 2 tuples of integers indicating
        the number of integers to cut off on the left and the number to cut off on the right
        
        For example:
            {"Journal of Magic":(0,3),
             "Journal of Unicorn Medicine":(1,0)}
        
        would indicate that for the Journal of Magic plot, the 3 most recent years
        (on the right) should not be plotted, while for Journal of Unicorn Medicine,
        the first year on the left should not be plotted
        
        this might be useful, for example, if data for year 2021 were only partial
        so didn't want to plot them...since groups may cover different year ranges,
        zoom_dict is used to allow flexibility

    For plot of single topic:
    -------------------------
    
        detect_max_val : bool, optional
            if True, detects maximum size value occuring anywhere for any topic
            and sets y axis upper limit to 1.1 * this. Use this any time you want
            to be able to compare plots
    
        title_group_label : str, optional 
            if custom_title is specified, this is irrelevant
            otherwise, if specified, this is used in title of plot, which is 
            "Topic Size Over Time by %s" % title_group_label
            if not specified, this is set to "Group"

        custom_title : str, optional
            custom title for the plot


        set_figsize : bool, optional
            if False, figure size is not set. Default is true
            
        figsize : tuple, optional
            if set_figsize = True, figure size is set to this. Default is (15,5)
            
        legend_outside_plot : bool, optional
            if True, legend is moved outside of the plot. Default is False
            
        legend_loc : tuple of two values, optional
            if legend_outside_plot = True, this tuple specifies where to plot it
            default upper right (1,1)
            
        legend_label : str, optional
            if given, this is the title of the legend
     
    For grid with multiple topics:
    ------------------------------    
        title_group_label and custom_title play same role as above
            only now for suptitle of entire grid. Titles of individual
            plots are automatically set to "Topic <topic_id>"
            
        figsize, set_figsize, legend_outside_plot have no effect here
     
        legend_label is not available here for space reasons. Recommended
        to use title_group_label or custom_title to make it clear what groups
        represent
   
        note: detect_max_val has no effect here because it is handled automatically 
            all grid plots will have same y axis
    
    
    saving options
    --------------
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

        if plotting grids, fig_name is appended with a # for each grid and each
        grid is saved separately
    
    Returns
    -------
    dictionary (one of the ones output by _get_topic_by_group_time_values())
          keys are group names
          each value is itself a dictionary with years as keys and topic size
               for topics 0...K (in order) for each year for that group
               (where K = theta_mat.shape[0])
     

    """
    #some checks
    K = theta_mat.shape[0]
    if type(to_plot_list) == list:
        assert np.all([type(val) == int and val >= 0 and val < K for val in to_plot_list]), "Invalid topic ids in to_plot_list"
    else:
        assert to_plot_list == "all"
        to_plot_list = list(range(K))
    if zoom_dict is not None:
        assert type(zoom_dict) == dict, "zoom_dict must be a dictionary"
        assert np.all([type(val) == tuple and len(val) ==2 for val in list(zoom_dict.values())]), "zoom_dict must contain length 2 tuples"
    
    
    per_group_year_dicts, unique_year_dict, groups = _get_topic_by_group_time_values(theta_mat = theta_mat,
                                                                                    group_list = group_list,
                                                                                    year_list = year_list,
                                                                                    doc_lengths = doc_lengths,
                                                                                    sizetype = sizetype)
    
    #some plotting preparations
    ylab = "Normalized Topic Size\n(%s)" % sizetype.replace("_"," ")
    if title_group_label is None:
        title_group_label = "Group"
    max_val= np.max([Helpers._get_max_size_in_dict(per_group_year_dicts[g]) for g in groups])
        
    
    
    #single plot
    #----------------------------------------------------------------------------
    if len(to_plot_list) == 1:
        if not detect_max_val:
            max_val = None
        else:
            max_val = max_val * 1.1
            
        group_dict = _get_topic_dict(topic_id = to_plot_list[0],
                                 groups = groups,
                                 per_group_year_dicts = per_group_year_dicts,
                                 unique_year_dict = unique_year_dict)
        
        if custom_title is None:
            custom_title = "Topic %d: Size Over Time by %s" % (to_plot_list[0], title_group_label)
        
        _group_over_time_plotter(group_dict = group_dict,
                                 year_dict = unique_year_dict, 
                                zoom_dict = zoom_dict,
                                title_group_label = title_group_label, 
                                title = custom_title, 
                                set_figsize = set_figsize,
                                figsize = figsize,
                                ylabel = ylab,
                                plot_legend = True,
                                legend_outside_plot = legend_outside_plot,
                                legend_label = legend_label,
                                legend_loc = legend_loc,
                                max_y_val = max_val)  
        
        if save_fig:
                figure_saver(fig_name = fig_name, 
                             outpath = fig_outpath,
                             dpi = dpi,
                             fig_override = fig_override,
                             bbox_inches = "tight")
                
                
    #create grid if asked for more than one plot
    #in this case there is no option to adjust overall figure size
    #----------------------------------------------------------------------------
    else:
        grids_list = Helpers.get_grids(to_plot_list, num_col = 4, num_row = 5)
        
        #create all the grids
        for p, plot_list in enumerate(grids_list):
            num_plots = len(plot_list)
            nrows, ncols = Helpers.figure_out_grid_size(num_plots = num_plots, 
                                                      num_cols = 4,
                                                      max_rows = 5,
                                                      adjust_for_low_num_plots = True)
            
            #deal with annoying adjustements to make plot look good for varying sizes of grid
            if num_plots <= 4:
                if num_plots >= 3:
                    fontsize = 20; top = 0.72; titlesize = 26; legend_loc = (.33,.83); legendsize = 16; xheight = -.06
                elif num_plots == 2:
                    fontsize = 18; top = 0.48; titlesize = 26; legend_loc = (.35,.75); legendsize = 16; xheight = -.13
                elif num_plots == 1:
                    fontsize = 16; top = 0.75; titlesize = 26; legendsize = 16; xheight = -.10
            else:
                fontsize = 35; legendsize = 18; titlesize = 30
                if nrows == 2:
                    top = 0.80; legend_loc = (.75,.89); xheight = -.06
                elif nrows == 3:
                    top = 0.83; legend_loc = (.75,.91); xheight = -.06
                elif nrows == 4:
                    top = 0.88; legend_loc = (.75,.95);  xheight = -.03
                elif nrows == 5:
                    top = 0.90; legend_loc = (.75,.95); xheight = -.03

            #figure size heuristic
            figsize = (5*ncols, 6*nrows)
            
            #set-up grid
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize= figsize)
            fig.text(0.5, xheight,  "Year", ha='center', fontsize = fontsize)
            fig.text(-.07, 0.4, ylab, va='center', rotation='vertical', fontsize = fontsize)
            fig.tight_layout(h_pad= 10, w_pad = 6) 
            
            #main plotting
            for i in range(ncols * nrows):
                ax = plt.subplot(nrows,ncols,i+1)
                if i < num_plots:
                    group_dict = _get_topic_dict(topic_id = plot_list[i],
                                 groups = groups,
                                 per_group_year_dicts = per_group_year_dicts,
                                 unique_year_dict = unique_year_dict)
                    
                    #turn some things off because gridding
                    _group_over_time_plotter(group_dict = group_dict, 
                                            year_dict = unique_year_dict, 
                                            zoom_dict = zoom_dict,
                                            set_figsize = False, 
                                            ylabel = ylab,
                                            title = "Topic %d" % plot_list[i],
                                            plot_legend = False,
                                            set_ylabel = False,
                                            set_xlabel = False,
                                            max_y_val = max_val * 1.1)
                    #get information for legend
                    handles, labels = ax.get_legend_handles_labels() 
                else:
                    #turn off any plots that don't get filled
                    ax.set_visible(False)
            
            #create legend
            fig.legend(handles, labels, loc = legend_loc, fontsize = legendsize)

            #overall title:
            if custom_title is None:   
                custom_title = "Topic Size Over Time by %s" % title_group_label
            plt.suptitle(custom_title, fontsize = titlesize)
            plt.subplots_adjust(top = top)  
            if save_fig:
                figure_saver(fig_name = fig_name + str(p), 
                             outpath = fig_outpath,
                             dpi = dpi,
                             fig_override = fig_override,
                             bbox_inches = "tight")

    
    return(per_group_year_dicts)






#plot word plot and topic over time plot by group on one grid
#similar to plot_barplot_and_timeplot() function in LdaOutputTimePlots.py

                       
                        
                        
                        
                    
def plot_barplot_and_grouped_timeplot(model, topic_id, corpus, dictionary, #core barplot components
                              theta_mat, group_list, year_list, doc_lengths, sizetype, #core timeplot components
                              zoom_dict = None,  legend_label = None, title_group_label = None, #additional timeplot stuff
                              value_type = "counts",topn = 20, lamb = 0.6, #additional barplot stuff
                              detect_max_val = True, plot_suptitle = True, custom_title = None,  #general stuff                            
                              figsize = (14,10),
                              save_fig = False, fig_outpath = None, 
                              fig_name = "bar_and_timeplot", dpi = 200,
                              fig_override = False): 
    
    """    
    Parameters
    ----------
    
    overall
    -------
        topic_id : int
            topic to plot
            
        figsize : tuple, optional
            adjusts size over overall grid, default (14,10)
    
        plot_suptitle : if False, plots no overall title
        
        custom_title : Optionally, specify your own overall title
            if not specified, title is "Topic <topic_id> from <model.num_topics>-Topic Model"
            
        detect_max_vals : bool
    
            if True, scales each plots relative to all topics in the model
            
            That is, looks at over time values to find the maximum value that
            occurs and sets y axis of time plot accordingly.
            Similarly, looks at all magnitudes used in bar plots and sets
            axis accordingly.
            
            Do this to make plots for multiple topics easier to compare and
            to avoid being misled by axis differences
    
    
    
    barplot parameters
    ------------------
        model : gensim LDA model
  
        corpus : gensim corpus
            corpus used to train LDA models
            (vector-count representation)
      
        dictionary : gensim dictinary 
            dictionary used to train LDA models
   
        topn : number of words to display
        
        lamb : lambda to use in relevance calculations
        
        value_type : whether to plot expected counts or fitted probabilites

   
   
    timeplot parameters
    -------------------
        theta_mat : overall theta matrix as output by get_document_matrices
        
        group_list : list of group labels for each document 
    
        year_list : list of year labels for each document
            
        doc_lengths : if sizetype is "word_count", then this must be specified: a list of
            the number of words in each document as output by LdaOutput.get_doc_lengths()
          
        sizetype : str
            one of "word_count", "doc_count", "mean"
            specifies what measure of topic size to use
      
        zoom_dict : optional dictionary for zooming in on parts of the graph
            keys are group names and values are length 2 tuples of integers indicating
            the number of integers to cut off on the left and the number to cut off on the right
            
            For example:
                {"Journal of Magic":(0,3),
                 "Journal of Unicorn Medicine":(1,0)}
            
            would indicate that for the Journal of Magic plot, the 3 most recent years
            (on the right) should not be plotted, while for Journal of Unicorn Medicine,
            the first year on the left should not be plotted
            
            this might be useful, for example, if data for year 2021 were only partial
            so didn't want to plot them...since groups may cover different year ranges,
            zoom_dict is used to allow flexibility
        
       legend_label : str, optional
            if given, this is the title of the legend
        
        title_group_label : str, optional 
            this is used in title of plot, which is 
            "Topic Size Over Time by %s" % title_group_label
            if not specified, this is set to "Group"


        
    save_fig...fig_override are the standard figure saving options. See 
        fig_saver() function in Helpers.py

    
    Returns
    -------
    None.

    """
    K = model.num_topics
    assert theta_mat.shape[0] == K, "Model and theta_mat do not have same number of topics"
    assert topic_id < K, "topic_id must be less than K = %d" %K  
    
    
    fig = plt.figure(figsize = figsize)
    gs = gridspec.GridSpec(30,25)
    
      
    #over time plot
    fig.add_subplot(gs[0:19,13:30])
    _ = topic_by_group_time_plot(theta_mat,
                   group_list = group_list,
                   year_list = year_list,
                   zoom_dict = zoom_dict,
                   doc_lengths = doc_lengths,
                   sizetype = sizetype,
                   to_plot_list = [topic_id],
                   title_group_label = title_group_label,
                   legend_label = legend_label,
                   legend_loc = (0.1,-.12),
                   detect_max_val = detect_max_val,
                   legend_outside_plot = True,          
                   set_figsize = False,
                   save_fig = False)
    
  
    #barplot
    fig.add_subplot(gs[0:25,0:10])
    LdaOutputWordPlots.topic_relevance_barplot(model = model, 
                                                topicid = topic_id, 
                                                corpus = corpus, 
                                                dictionary = dictionary,
                                                lamb = lamb,
                                                value_type = value_type, 
                                                theta_mat = theta_mat, 
                                                topn = topn, 
                                                detect_max_x = detect_max_val,
                                                plot_title = True,
                                                title = "%d most relevant words ($\lambda$ = %s)" % (topn, str(lamb)),
                                                set_figsize = False, 
                                                save_fig = False,
                                                )
    #overall title:
    if plot_suptitle:
         if custom_title is None: 
             custom_title = "Topic %d from %d-Topic Model" % (topic_id, K)
         plt.suptitle(custom_title, fontsize = 30)
         plt.subplots_adjust(top= .85)    

    if save_fig:
            figure_saver(fig_name = fig_name, 
                         outpath = fig_outpath,
                         dpi = dpi,
                         fig_override = fig_override,
                         bbox_inches = "tight")
    




















