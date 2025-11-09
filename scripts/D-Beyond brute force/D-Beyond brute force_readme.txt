This folder follows the stochastic brute force approach, by enriching or compressing the materials database.

The stochastic brute-force initial database is purely based on a quinary list of elements that should lead to HEA when mixed together. Therefore, the materials that are generated are localized in a tiny part of the global space, which may be detrimental to finding original compositions and training the CVAE. To complete the database, it is possible to use the Hamming distance to cover the space of possibilities with at least one material within a given distance. The script 1-hamming_glouton.py generated such a set of materials given the parameters (in particular the given distance). Parameters must be adjusted (lines 7 to 25).

Different scripts, and mostly the CVAE script, may eventually generate very close or identical compositions, which causes several problems:
1. The set of generated materials is uselessly long to study
2. The number of entries do not represent the diversity that was obtained
3. Very close materials cause a bias in the statistical analysis of the generated materials.
To answer this, the script Compress_Pareto_Ward.py compresses a database according to the Ward distance (within a tiny cluster, only the medoid remains). Adjust parameters (input / output files, thinness of clusters) at lines 28 to 36.

It may also be useful to work temporarily with a significantly narrowed database (to run tests for instance). In that case, a similar script is applied. To handle very large entries, it splits the database in parts, compress them, merge them, and so on. This script is Divide_Conquer.py. Adapt parameters (file names and compression 'THRESHOLD') at lines 16 to 19.

Instead of the very basic stochastic brute force, it is possible to use the fact that 5 targets are differentiable to operate a gradient ascent to generate new compositions from initial entries. This is the aim of Gradient_Augmentation.py script. When given a set of elements, it performs a gradient ascent starting from the different isostoechiometric quinaries to optimize the 5 targets. It saves the best candidates (according to the log-sum of the targets) in an output file to be specified line 17.
The same idea can enclose all 10 targets, but since the XGBoost surrogates (and the calculation of certain descriptors that are based on libraries like enthalpy!) are not differenciable, it is achieved through a basic "finite difference method", in simplex_steepest_ascent.py. The input file containing the compositions to be improved,'pareto_csv', must be specified line 22, and the output file line 23.

When the time comes to train the CVAE, the available database may contain few materials with fine properties. To help the algorithm for the training phase, it is possible to augment the set of good materials by creating "noise" on their composition. This is the point of the script augment_pareto.py. Input and output files are to be specified at lines 13 and 14; parameters describing the noise are defined at lines 21 to 30.
