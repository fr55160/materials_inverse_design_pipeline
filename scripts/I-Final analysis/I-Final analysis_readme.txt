This folder contains the tools to analyze the final set of candidates.

1-pareto_clusters_ward.py realizes the ward clustering of the overall Pareto front, using the elbow method to choose the number of clusters. The source file containing the Pareto front must be specified line 13 (variable 'IN_CSV').

2-properties_by_cluster_ward.py generates bar graphs for 7 targets (Melting Point (K), Creep (Larson–Miller parameter), High-T Oxidation (log $K_A$), Density, Bulk Modulus (GPa), Shear Modulus (GPa), G/B ratio (Pugh)) representing the average value & standard deviation obtained for each cluster (Following the Ward clustering).

