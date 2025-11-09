This folder is dedicated to the enrichment of the raw database: from the normalized composition of the materials, compute de different compositional descriptors using Descriptors_Hephaistos.py.

First, adapt 1-Generate_Descriptors_HT_Oxidation.py to the database you need to enrich.
Adapt the values of variables 'source' and 'outfile' (lines 21 and 22) to treat to raw data coming from different database.

Second, compute the outlier score using 2-Outliers_Detector.py.
Adapt the values of variables 'target1_source' to 'target2_source' with the names of the previous 'outfiles', so as to treat each database.

Eventually, each database is enriched with the compositional descriptors (42) and one outlier score column per "target" (quantity to learn to predict).