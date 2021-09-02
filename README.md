## Information: Partial Public Repo
Copyright © 2021 Kyla E. Chasalow
kec89@cornell.edu

This repo contains code and plots for my 2021 Oxford MSc in Statistical Science dissertation, "Reading by Other Means:  Exploring Social Science Texts with Topic Models" [not yet available, as it still needs to be marked for my degree]. The data used for this project were downloaded from the Scopus API in May 2021 via http://api.elsevier.com and http://www.scopus.com. Specifically, the data include all articles in the Scopus database for the selected journals  (Demography, American Journal of Sociology, American Sociological Review, Annual Review of Sociology) up to May 18, 2021. Data are available upon request to academic authors with approved access to the Scopus API. Due to Scopus use policies, it cannot be posted publicly. The trained models are, at this time, also not posted publicly but can, if one has access to the data, be replicated with random state in gensim set to 175. All this means that the scripts in \src\scripts will not actually run. The functions in \src\func should still be applicable for training gensim LDA models and analyzing their output. EdaHelper.py is the one function in that directory that is more specific to the data but it will work for Scopus data in general. In \src\demos, the modelling demo uses a toy example. The plotting demo currently uses the data and thus will not run but I plan to add a demo that uses some publicly available example in the future. 

See **license** file for licensing information

See **requirements.txt** for conda environment file

See **ReadMe.pdf** file in src for details on code 

--------------------------------
**Table of Contents**

* **src** contains code. Within it, **func** subfolder contains only scripts with functions. **scripts** subfolder contains scripts for running analysis, starting with Demog for Demography analysis and Socio for sociology analysis. **demos** contains demonstration notebooks.

* **results** is for model and plot results from the analysis. Note that models are not included at this time.

  

**Note** Empty **data** and **models** folders are included to show where those outputs *would be* stored.



-----

**Planned Future Additions**

* Generalize code to allow not just analyses of documents by year but also analyses  by hours, days, months etc. -- or perhaps just any ordered variable. For my thesis, the analysis was by year, so some of the code assumes that the time labels given are years. I plan to generalize this.
* Make LdaLogging.py functions for analyzing logs a bit more robust -- they should recognize when the log they are given does not contain the input needed to do what they are asked to do
* Add a plotting demo notebook that uses open source data 
