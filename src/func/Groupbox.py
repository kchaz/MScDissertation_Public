# -*- coding: utf-8 -*-
"""

This file contains a general function for plotting grouped box plots


Author: Kyla Chasalow
Last edited: August 10, 2021


some inspiration drawn from
https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots


"""

import matplotlib.pyplot as plt
import numpy as np
import numpy_indexed as npi



# function for setting the colors of the box plots
def _setBoxcolor_list(bp, color_list, alpha, fill_fliers = True):
    """
        
    Helper function to set color of a group of box plots
    sets boxplots in bp to color_list except for median line, which is kept black
    
    whether or not boxplots are filled in depends on whether 
    bp was created with patch_artist = True or False
    
    
    Parameters
    ----------
    bp : box plot object as output by call to plt.boxplot()
    color_list : list of strings (color_list) for each boxplot plotted by bp object
    alpha : float, transparency of boxplots
    fill_fliers : bool, optional
        if True, fliers are filled with color. Else, only outline is color and
        filling is default white. The default is True.

    Returns
    -------
    None.

    """
    for i, col in enumerate(color_list):
        plt.setp(bp["boxes"][i], color = col, alpha = alpha)
        plt.setp(bp['medians'][i], color = "black")
        if fill_fliers:
            plt.setp(bp["fliers"][i], markeredgecolor = col,
                     alpha = alpha, markerfacecolor=col)
        else:
            plt.setp(bp["fliers"][i], markeredgecolor = col,
                         alpha = alpha) 
        
    #things that appear twice on each boxplot
    doubles = np.arange(0,2*len(color_list),2) 
    for i, col in enumerate(color_list):
        plt.setp(bp["caps"][doubles[i]], color = col, alpha = alpha)
        plt.setp(bp["caps"][doubles[i]+1], color = col, alpha = alpha)
        plt.setp(bp["whiskers"][doubles[i]], color = col, alpha = alpha)
        plt.setp(bp["whiskers"][doubles[i]+1], color = col, alpha = alpha)      



    
def grouped_boxplots(groups, group_labels, 
                     color_list = None, color_labels = None, alpha = 0.5, 
                     fill_boxes = True, fill_fliers = True,
                     vert = False, space_between = 1,
                     title = None, xlabel = None, ylabel = None, 
                     set_figsize = True, figsize = (15,5),
                     xlim = None, ylim = None,
                     xticks_rotation = 0, yticks_rotation = 0):
    """
    plot grouped boxplots, vertically or horizontally


    Parameters
    ----------
    groups : list of lists of lists of values
        each entry of groups represents a group of values for each of a number of levels
        see example below. 
        
        Note: it is fine for some levels for some groups to be empty lists
        but there must be the same number of levels in each entry in groups
            
        
    group_labels : labels for each cluster of boxplots, same length as groups
    
    color_list : colors for within each cluster of boxplots (e.g. if clusters of 3, need 3 colors)

    color_labels : labels for each boxplot within cluster of boxplots (used in legend)

    alpha : float, optional
        transparency of boxplots. The default is 0.5.
   
    fill_boxes : bool, optional
        if True, boxplots are filled in. The default is True.
        
    fill_fliers : bool, optional
        if True, fliers (outlier points greater in abs value than 1.5*IQR) on boxplot 
        are filled in. The default is True.
        
    vert : bool, optional
        if True, boxplots are plotted vertically. The default is False.
        if False, boxplots are plotted horizontally
        
    space_between : int, optional
        number of spaces to leave between clusters of boxplots. The default is 1.
    
    title...yticks_rotation are standard plotting options that set the title, ylabel,
    xlabel, axis limits, and tick rotation of the plot. There is an option not to set
    the figure size (set_figsize = False) which may be necessary if using this plot in a grid
   
    Returns
    -------
    None.



    Example
    --------
    A= [np.array([1, 2, 5,]),  np.array([])]   #note empty lists and numpy arrays are ok
    B = [[7, 2, 5, 10, 11, 10, 25],[3,4,5]] 
    C = [[-10, 3,2,5,7], [4, 6, 7, 3]]
    
    groups = [A, B, C]
    labels = ["A","B","C"]
    
    
    grouped_boxplots(groups = groups,
                     group_labels = labels,
                     color_labels = ["Type 1","Type 2"],
                     colors = ["red","blue"],
                     alpha = 0.3,
                     fill_boxes = True,
                     fill_fliers = True,
                     vert = False,
                     space_between = 1,
                     title = "This is my title",
                     xlabel = "X axis",
                     ylabel = "Y axis",
                     set_figsize = True,
                     figsize = (10,5))
   
    """
    assert type(space_between) == int and space_between >= 0, "space_between must be an int >= 0"
    


    if color_list is not None:
        assert color_labels is not None, "If color_list given, color_labels must not be None"
    
    if set_figsize:
        plt.figure(figsize = figsize)
    #ax = plt.axes()

    #figure out boxplot positions
    num_outer_groups = len(groups)
    num_inner_groups = len(groups[0]) 
    interval = num_inner_groups + space_between
    position_list = [list(range(i,i+num_inner_groups,1)) for i in np.arange(1,interval*num_outer_groups, interval)]
   
    for i, g in enumerate(groups):
        bp = plt.boxplot(g, positions = (position_list[i]), widths = 0.6, vert = vert,
                         patch_artist = fill_boxes, 
                         boxprops=dict(alpha = alpha),
                         )        
        if color_list is not None:
            _setBoxcolor_list(bp, 
                          color_list = color_list,
                          alpha = alpha, 
                          fill_fliers = fill_fliers)
            
    ticks = [np.mean(elem) for elem in position_list]
    if vert:
        plt.yticks(fontsize = 15, rotation = yticks_rotation)
        plt.xticks(ticks = ticks, labels = group_labels, fontsize = 16, rotation = xticks_rotation)
    else:
        plt.yticks(ticks = ticks, labels = group_labels, fontsize = 16, rotation = yticks_rotation)
        plt.xticks(fontsize = 15, rotation = xticks_rotation)
        
    if title is not None:
        plt.title(title, pad = 20, fontsize = 25)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize = 18)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize = 18)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
            
    # draw temporary lines and use them to create a legend
    if color_list is not None:
        line_list = []
        for col in color_list:
            h, = plt.plot([1,1],col, alpha = alpha, linewidth = 4)
            line_list.append(h)
            
        plt.legend(line_list, color_labels, fontsize = 16 )
        for h in line_list:
            h.set_visible(False)




### Important and Useful Functions for Grouping

def group_values_by_two_dict(outer_variable, inner_variable, values):
    """
    obtain a dictionary of dictionaries grouping values by levels of two variables
    the unique values of outer_variable form the keys of the overall dictionary output
    but the function. The unique values of inner_variable form the keys of each the
    dictionary that is the value for each of those outer keys. Each inner dictionary
    has the same keys (though they may not be entered in the same order). For each
    inner dictionary key, the values for that level of inner dictionary and the given
    level of outer_variable is given
    
    For example, outer_variable might be years and inner_variable might be journal
    titles and values might be abstract lengths. IN that case the dictionary output by this
    function would have a key for each year, and for each year, there would be a dictionary
    with the abstract lengths for the observations for each journal
    
        

    Parameters
    ----------
    outer_variable : list or array of strings, integers, or floats
    inner_variable : list or array of strings, integers, floats
    values : list or array of strings, integers, floats

    * all three must have same length *

    Returns
    -------
    dictionary as described above
    
    
    Example
    -------
    >>> ears = [2013, 2013, 2013, 2014, 2014, 2014, 2015, 2015]
    >>> colors = ["Blue","Blue","Green","Green","Blue","Blue","Green","Green"]
    >>> counts = [1,5,9,1,3,4,2,8]
    >>> group_values_by_two_dict(outer_variable = years,
                             inner_variable = colors,
                             values = counts)


    {2013: {'Blue': array([1, 5]), 'Green': array([9])},
     2014: {'Blue': array([3, 4]), 'Green': array([1])},
     2015: {'Green': array([2, 8]), 'Blue': []}}
        

    """
    assert len(outer_variable) == len(inner_variable), "outer_variable and inner_variable must have same length"
    assert len(outer_variable) == len(values), "outer_variable and values must have same length"
    
    #used make sure each dictionary has same keys, even if some levels of outer do not have
    #values for all levels of inner
    unique_inner = set(np.unique(inner_variable))
    
    #group values by outer and inner variable
    outer_labels1, grouped_values = npi.group_by(keys = outer_variable, values = values)
    outer_labels2, grouped_inner = npi.group_by(keys = outer_variable, values = inner_variable)
    assert np.all(outer_labels1 == outer_labels2), "internal sanity check 1" 
    
    out_dict = {}
    for i,y in enumerate(outer_labels1):
        #get the values grouped by inner variable for each level of outervariable
        inner_labels, inner_value_groups = npi.group_by(keys = grouped_inner[i], values = grouped_values[i])
        #add the vector of values for each inner variable level by name to summary dictionary
        summary_dict = {}
        for j, level in enumerate(inner_labels):
            summary_dict[level] = inner_value_groups[j]

        #check for any levels of inner variable that aren't present for this level of outer
        #variable and add empty list for them
        excluded = unique_inner.difference(set(summary_dict.keys()))
        for e in excluded:
            summary_dict[e] = []
        out_dict[y] = summary_dict
        
    return(out_dict)





def convert_dict_to_lists(d):
    """
    collapses dictionary as output by group_values_by_two_dict into list
    of lists of lists as required for input into grouped_boxplots function

    Parameters
    ----------
    d : dictionary as output by group_values_by_two_dict

    Returns
    -------
    0. list of lists of lists (or arrays) - collapsed dictionary
    1. outer labels - the outer keys of d
    2. inner labels - the inner keys of each entry of d (assumed to be same for each entry)

    * Note that function is careful to make sure that values in each inner list
    are in same order as inner labels, even if for dictionary, they are not


    Example
    ..........
    >>> years = [2013, 2013, 2013, 2014, 2014, 2014, 2015, 2015]
    >>> colors = ["Blue","Blue","Green","Green","Blue","Blue","Green","Green"]
    >>> counts = [1,5,9,1,3,4,2,8]
    >>> d = group_values_by_two_dict(outer_variable = years,
                             inner_variable = colors,
                             values = counts)
    
    >>> groups, outer_labels, inner_labels = convert_dict_to_lists(d)
    >>> print(groups)
    
    [[array([1, 5]), array([9])],
     [array([3, 4]), array([1])],
     [[], array([2, 8])]]

    >>> print(outer_labels)

    [2013, 2014, 2015]

    >>> print(inner_labels)

    ['Blue', 'Green']

    """
   
    outer_keys = list(d.keys())
    
    #overall holder
    out_list = [None] * len(outer_keys) 
    
    #assuming all have same keys - group_values_by_two_dict ensures this
    #defining this at start will ensure that entries in each nested list are in same inner label order,
    #even if not that way in dictionary (which could happen with exclusion add-ons bit that ensures
    #all dict have same label -- see group_values_by_dict())
    inner_keys = list(d[outer_keys[0]].keys()) 
    for i,k in enumerate(outer_keys):
        nested_dict = d[k]
        out_list[i] = [nested_dict[j] for j in inner_keys]
    
    return(out_list, outer_keys, inner_keys)







