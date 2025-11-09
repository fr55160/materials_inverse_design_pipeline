This folder contains scripts to analyze the database of materials.

The script ACP_Pareto_compositions_CLR.py is dedicated to the analysis of a Pareto front, and more specifically, to the chemical composition of its materials. It generates a graph whose vertices are the chemical elements, and the edges carry the "importance" of the correlation between pairs of elements A and B, defined as 1-corr(A,B). The file containing the set of alloys must be specified line 15. Since this script was developped to treat Pareto fronts, for which 1 column per element was generated, this input file must be structured correspondingly.

The visualization of the Pareto front can be done on Ashby graphs through Analysis_and_plots_HEA_2.py. The input file must be specified line 20.
Important warning: the calculation of the supply risk for the Ashby graph in this script relies on an approximation (the correlation between elements is neglected). A more rigorous approach is adopted with the scripts in G-risks.