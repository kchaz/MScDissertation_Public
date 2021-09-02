# -*- coding: utf-8 -*-
"""

This file contains a general scatterbox function which I use extensively
for visualizing the spread of various per-topic metrics across different
LDA topic models. The code is written to be general, however. 

Author: Kyla Chasalow
Last edited: July 12, 2021


"""
import matplotlib.pyplot as plt
import numpy as np

### HELPERS

#Helpers are used to deal with issue that boxplot plots outliers as points
#and if you then overlay boxplot with points while adding random jitter,
#you get problem that it shows the outlier points twice. These functions together
#generate those x values needed to plot the points while not adding jitter
#for x values corresponding to the outlier points

def _generate_x(arr, i, ignore, noise = 0.05):
    """
      generate array of random noise around number i of length len(arr)
    with all elements equal to elements in ignore set equal to i 
    That is, don't add jitter to array values corresponding to values in ignore
    
    meant for plotting points overlaying a boxplot with jitter so can see all the points.
    e.g. to plot points vertically for boxplot centered at location 1, 
    need an array of 1's + jitter corresponding to array of values
    Don't want to jitter the 'fliers' because boxplot already plots those, so this function keeps those as just 1
    
    e.g. if arr = [-20,1,2,4,5,50,3,50]  and fliers = np.array([-20,50]) and i = 1
    then output should be something like [1, 0.98, .95, 1.01, .999, 1, .987, 1] where the
    non-1 values are randomly generated
    

    Parameters
    ----------
    arr : list or numpy array of floats or integers
        
    i : int or float
        noise will be centered around this value and 
        elements of array that are in ignore will get this value 
        without noise added in the output array
        
    ignore : list or array of ints or floats, all of which occur at least once 
        in arr. These are the values in arr that are not to recieve noise in 
        output array

    noise : float, optional
        amount of normal random noise to add. The default is 0.05.

    Returns
    -------
    numpy array of random draws from normal centered at i
    EXCEPT that for entries of output array that correspond to entries in arr
    which contained values from ignore, the output array contains i without
    random noise

    """
    assert type(arr) == list or type(arr) == np.ndarray, "input must be list or numpy array"    
    
    if type(arr) == list:
        arr = np.array(arr) 
    
    x = np.random.normal(i, noise, size = len(arr))
    
    #if no fliers, just return x
    if list(ignore) == []: 
        return(x)
    
    #get indices of arr corresponding to those to fliers
    rm_ind = np.concatenate([np.where(arr == elem)[0] for elem in ignore]) 
    x[rm_ind] = i
    return(x)



def _get_x_array_for_boxplot_points(arr_list, fliers, noise = 0.05):
    """ apply _generate_x function to all elements of arr_list (a list of arrays)
    with ignore list for each array in arr_list contained in fliers
    
    the i+1 is because box plots are numbered starting at 1 so noise for first
    needs to center at 1
    """
    assert len(arr_list) == len(fliers), "array list and fliers must have same length"
    out = [_generate_x(arr_list[i],i+1,fliers[i], noise = noise) for i in range(len(fliers))]
    return(out)
 
    

### MAIN FUNCTION
    
def scatterbox(arr_list,  labels, plot_points = False, title = "",
                          color_box = "purple", color_point = "blue", xlabel = "", ylabel = "",
                          alpha_box = .3, alpha_point = .3, ylim = None,
                          set_fig_size = True, figsize = (12,7), plot_legend = True,
                          legend_point_label = " ", xtick_rotation = 0, ytick_rotation = 0):
    """
    
    Plot vertical side-by-side boxplots, possibly overlayed with points


    Parameters
    ----------
    arr_list : a list of lists or np.arrays 
        each element contains the values to be used 
        for each boxplot.
        
    labels : list of strings
        labels for the box plots
        must be of same length as arr_list
        
    plot_points : Bool, optional
        optionally, plot points in addtion to boxplots.
        The default is False.
        
    title : str, optional
        plot title. The default is "".
        
    color_box : str, optional
        boxplot color. The default is "purple".
        
    color_point : str, optional
        point color. The default is "blue".
        
    xlabel : str, optional
        x axis label. The default is "".
        
    ylabel : str, optional
        y axis label. The default is "".
        
    alpha_box : float, optional
        boxplot transparency. The default is .3.
        
    alpha_point : float, optional
        point transparency. The default is .3.
        
    ylim : tuple of ints, optional
        y axis limits, of form (#,#). The default is None.
                                
    set_fig_Size : bool, optional, default is True
        in general, function will set a figure size (either by default)
        or figsize argument) if plotting these box plots as part of a grid
        need to turn this off so that figsize setting doesn't conflict
        with grid settings. 
        
    figsize : tuple of two values, default is (12,7)
        controls dimensions of resulting plot, as long as set_fig_size = True
        
    plot_legend : bool, default is True
        if true, plots legend with labels as given in labels argument
        
    legend_point_label : str, default is " "
        the label to plot as a description of what the points in the scatterbox
        represent
        
    xtick_rotation : int, default is 0
        rotation for x ticks
        
    ytick_rotation : int, default is 0
        rotation for y ticks

    Returns
    -------
    None.

    """
    #text options 
    plt.rcParams.update({'font.family':'serif'})
    
    #box plot
    if set_fig_size:
        plt.figure(figsize = figsize)
        
    box = plt.boxplot(arr_list, vert=True, labels=labels, 
                            patch_artist = True,
                            boxprops=dict(facecolor=color_box,
                                          color=color_box, alpha = alpha_box),
                            medianprops=dict(color= "black"), zorder = 0) 
    
    if plot_points:
        
        #get outlier points so can avoid adding jitter to them since boxplot already plots them
        fliers = [item.get_ydata() for item in box['fliers']]
        x = _get_x_array_for_boxplot_points(arr_list, fliers, noise = 0.05)
     
        plt.scatter(x[0],arr_list[0],
                   color = color_point, alpha = alpha_point, zorder = 2, s = 70,
                   label = legend_point_label)
        if plot_legend:
            plt.legend(fontsize = 17)
        
        for i in range(1,len(arr_list)):
            plt.scatter(x[i],arr_list[i],
                   color = color_point, alpha = alpha_point, zorder = 2, s = 70,
                   label = legend_point_label)

    #plot aesthetics
    plt.xlabel(xlabel, fontsize = 18)
    plt.ylabel(ylabel, fontsize = 18)
    plt.yticks(fontsize = 15, rotation = ytick_rotation)
    plt.xticks(fontsize = 15, rotation = xtick_rotation)
    if ylim != None:
        plt.ylim(ylim)
    plt.title(title, pad = 30, fontsize = 18)
    #not including plt.show() here so that can plot 
    #in suplots if desired
