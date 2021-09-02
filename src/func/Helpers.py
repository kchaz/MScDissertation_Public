# -*- coding: utf-8 -*-
"""
This file contains general use helpers

Author: Kyla Chasalow
Last edited: July 28, 2021

@author: kyla
"""

import matplotlib.pyplot as plt
import os
import numpy as np


### FILE SAVING 

def file_override_request(message = None):
    """
    prompts user about whether they want to override any existing files with same name
    in destination folder(s) and returns user input
    
    optionally specify a custom message

    """
    if message is None:
        override = input("Override existing files in destination folder(s) that have same name? (Yes/No):")
    else:
        override = input(message)
    assert override.lower() == "yes" or override.lower() == "no", "Invalid input: must be Yes or No"
    if override.lower() == "yes":
        override = True
    elif override.lower() == "no":
        override = False
    return(override)


def filename_resolver(filename, extension = None,  outpath = None, num_errors = 0):
    """
    
    Recursively, Tries to open file with name filename.extension If it already exists
    in working directory, tries amending filename with a number until
    it finds a name that doesn't currently exist
    
    Note that it doesn't actually add the extension to the filename it returns
  
    
    Example: 
        * suppose my_lda_model is a saved gensim model in working directory
        * calling filename_resolver("my_lda_model") will output
        "my_lda_model_1". If "my_lda_model_1" also already existed,
        would output "my_lda_model_2" etc.

    Parameters
    ----------
    filename : str
    
    extension : str, optional
        e.g. "txt" or "csv". Should not include the "." 
        The The default is None.

    outpath : str, optional, default None
        if outpath is specified, checks whether filename.extension exists
        in specified directory. If None, checks current working directory

    num_errors : int, optional
        For internal use in the recursion only. The default is 0.   

    Returns
    -------
    None.

    """
    #TYPE CHECKS
    assert type(filename) == str, "filename must be a string"
    assert type(num_errors) == int, "num_errors must be an int"
    if extension is not None:
        assert type(extension) == str, "extension must be a string but is type" + str(type(extension))
    if outpath is not None:
        assert type(outpath) == str, "outpath must be string or None"
    
    #add extension if given
    if extension is not None:
        try_open = filename + "." + extension
    else:
        try_open = filename
        
    #add outpath if given
    if outpath is not None:
        try_open = os.path.join(outpath, try_open)

    #try opening file        
    try:
        open(try_open, "r")
        file_problem = True #if successfully opens it, we have a problem
        num_errors += 1
    except Exception:
        file_problem = False
        
    #ammend file name and try the next version
    if file_problem:
        if num_errors == 1: #we're only at the first
            filename = filename + "_" + str(num_errors)
        else: #num_errors is greater than 1
            #find the last _ and get rid of it and anything after it so can ammend new _#   
            filename = filename[:filename.rfind("_")] + "_" + str(num_errors)
            
        filename = filename_resolver(filename,
                                     extension = extension,
                                     outpath = outpath,
                                     num_errors = num_errors)
    return(filename)




def figure_saver(fig_name, outpath = None, dpi = 400, bbox_inches = None, pad_inches = None,
                 fig_override = False):
    """
    generic function to save figure while using filename_resolver to make sure 
    any existing figures aren't overridden unless override is true

    Parameters
    ----------
    fig_name : str
        name of figure
        if does not include extension, default format is .png
        
    outpath : str, optional
        optionally specify where to save figure. Else saves in 
        working directory. The default is None.
        
    dpi : int, optional
        resolution in dots per inch. The default is 400.
        

    documentation for following two parameters for plt.savefig copied from here
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
    
    bbox_inches : str or Bbox, default: rcParams["savefig.bbox"] (default: None)
        Bounding box in inches: only the given portion of the figure is saved. 
        If 'tight', try to figure out the tight bbox of the figure.

    pad_inches : float, default: rcParams["savefig.pad_inches"] (default: 0.1)
        Amount of padding around the figure when bbox_inches is 'tight'.


    Returns
    -------
    None.

    """
    #check if already has a valid file extension
    split_name = fig_name.split(".")
    if len(split_name) == 2:
        if split_name[1] in ["eps", "jpeg", "jpg", "pdf", "pgf", "png",
                             "ps", "raw", "rgba", "svg", "svgz", "tif", "tiff"]:
            fig_name = split_name[0]
            extension = split_name[1]
        else:
            extension = "png"
    elif len(split_name) == 1: 
        extension = "png"  #default in savefig if no extension given - making that explicit here
    else:
        extension = "png"
        print("Warning: unable to detect extension due to presence of more than one `.` character. Saving as .png")
    
    #resolve filename if asked to
    if not fig_override:
        fig_name = filename_resolver(fig_name, 
                                     extension = extension, 
                                     outpath = outpath)
    
    if outpath is not None:
        filepath = os.path.join(outpath, fig_name + "." + extension)
    else:
        filepath = fig_name + "." + extension
        
    plt.savefig(fname = filepath,
                dpi = dpi,
                bbox_inches = bbox_inches,
                pad_inches = pad_inches)







#### PLOTTING HELP

def figure_out_grid_size(num_plots, num_cols, max_rows = None,
                         adjust_for_low_num_plots = True):
    """
    figures out how many rows and columns to plot in a subplot grid
    given you want to plot num_plot plots and you want there to be num_cols
    columns. Optionally specify a maximum number of rows. Then if hit maximum
    number of rows, will return max rows instead
    
    For example, num_plots = 20, num_cols = 4, max_rows = None returns (5, 4)
    but num_plots = 20, num_cols = 4, max_rows = 4 returns (4,4)

    adjust_for_low_num_plots:
        if True, takes special approach to 1, 2, 3, or 4 plots by 
        returning rows and columns for 1 x 1, 1 x 2 or 2 x 2 grid 
        
        For example, if num_plots = 4 and num_cols = 4, then if
        adjust_for_low_num_plots = False, get output (1,4)
        but if adjust_for_low_num_plots = True, get (2,2)
        which might look nicer

    Returns
    -------
    (number of rows, number of columns)

    """
    assert type(num_plots) == int and type(num_cols) == int, "both arguments must be ints"
    assert num_plots > 0 and num_cols > 0, "both arguments must be greater than 0"
    if num_cols > 8:
        print("Warning: you've asked me to create a larger than 8-column grid. That doesn't usually look very good")
    
    #handling the 4 or fewer cases - ignores num_cols and max_rows
    if adjust_for_low_num_plots:
        if num_plots <= 4:
            if num_plots == 1:
                nrows = 1; ncols = 1
            elif num_plots == 2:
                nrows = 1; ncols = 2
            else:
                nrows = 2; ncols = 2
            return(nrows, ncols)

    #general approach
    ncols = num_cols
    if max_rows is None:
        nrows = num_plots // ncols
        if num_plots % ncols != 0: # add a remainder row if there is one
            nrows += 1
    else:
        nrows = np.min([num_plots // ncols, max_rows])
        if nrows < max_rows: #add remainder if still have rows to spare
            if num_plots % ncols != 0:
                nrows += 1

    return(nrows, ncols)



assert figure_out_grid_size(20,4,None) == (5,4), "test of figure_out_grid_size"
assert figure_out_grid_size(20,4,4) == (4,4), "test of figure_out_grid_size"



def get_grids(to_plot_list, num_col, num_row):
    """Figure out how many grids of num_row x num_col are needed
    to create plot for every element of to_plot_list. If the number
    of plots does not divide evenly, last grid is a remainder grid
    with possibly fewer than num_row x num_col elements
    
    Output is a list of lists representing grids, with each list containing
    elements from to_plot_list to be plotted in that grid
    
    E.g. to_plot_list = [2,3,4,5,9,10] and num_col = 2
    would yield [[2,3,4,5],[9,10]]
    """
    
    N = len(to_plot_list)
    S = num_row * num_col
    num_full_grids = N // (S)
    remainder = N % (S)
    grids_list = []
    for i in range(num_full_grids):
        grids_list.append(to_plot_list[(i*S):(i*S+S)])
    if remainder != 0:
        grids_list.append(to_plot_list[len(to_plot_list)-remainder:])
    return(grids_list)





### MISCELLANEOUS TASKS


def _get_max_size_in_dict(dct):
    """For a dict containing lists/arrays of floats or ints as values, finds
    maximum value that occurs anywhere in dict
    """
    keys = list(dct.keys())
    get_all_maxes = [np.max(dct[key]) for key in keys]
    return(np.max(get_all_maxes))

def _get_max_size_in_dict_list(dct_list):
    """Gets maximum value from a list of dictionaries with
    lists/arrays of floats or ints as values """
    return(np.max([_get_max_size_in_dict(d) for d in dct_list]))
    
def _get_max_nested_list(lst):
    """Get the maximum value in a nested list"""
    return(np.max([entry for elem in lst for entry in elem]))
    