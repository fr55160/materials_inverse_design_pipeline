This folder is dedicated to the calculation of targets of given alloys, and the stochastic brute-force / analytic research of interesting alloys. 

The script 1-Descriptors_for_HEA.py takes a set of normalized composition (currently in "Alloys-list", to be adapted), computes the associated atomic descriptors, and using the pkl models, computes the predicted targets and associated scores.
Adapt the values of input / output files (lines 26-27).

WARNING: it is planned that after this step, the user adds MANUALLY a new column in the database indicating how the data was obtained:
* 'Brute Force HEA' if the stochastic brute-force method was employed to assign stoechiometries to a set of elements
* 'Hamming & Gradient Augmentation'  if it was obtained by generating a database paving the space regarding the Hamming-distance and adjusting the stoechiometries with a gradient on 5 differenciable scores.
* 'Annealing' if it was obtained by improving an existing database with the finite difference algorithm
* 'Generative CVAE' if it was obtained by generative CVAE process

The script 2-ward_linkage_clustering.py serves to cluster a material database based on the composition. Using the Ward linkage method, it adds a new cluster column to the database featuring the composition of the associated cluster medoid. It later serves the cluster analysis.
Adapt the values of input and output files (lines 30 and 31) : 'INPUT_CSV' and 'OUTPUT_CSV'. The 'THRESHOLD' variable indicates the threshold of the cut for the Ward linkage clustering. Adapt it to get more and less clusters.

The script 3-HEA_Pareto.py works from a database enriched with the targets and the associated scores. It keeps the materials reaching a threshold for every score (at least 0.3), and from this remaining set, it saves the Pareto front (materials with a non-dominated set of scores). This allows to identify the most promising candidates for a further analysis, but also provides interesting input for the training of the neural network of the CVAE. 
Adapt the values of 'file_path' and 'output_file' (lines 26-27).

