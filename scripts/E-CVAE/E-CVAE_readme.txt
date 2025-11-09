This folder is dedicated to the CVAE process to generate new candidates.

The script config_adaptative.py contains the different parameters to be fixed.

The script utils_io_adaptative.py contains different functions that may be called by other scripts: to parse the composition, to operate the CLR transformation, and so on.

The script preprocess_adaptative.py prepares the data for the training process; in particular, it creates the vector X (CLR composition, descriptors, targets) and Y (associated scores to the targets). This way, the model is fed with a continuous link from (CLR) compositions to targets.

The central script to train the CVAE and generate new candidates is evaluate_adaptative.py and the one to run. It realizes the different operations described in the flowchart of the manuscript, and uses subscripts :
- config_adaptative.py and utils_io_adaptative.py as explained before. Almost all parameters to be set are fixed in config_adaptative.py. In particular, the csv file containing the Learning database is specified line 28.
- scoring.py recalculates the "real" scores: indeed, scores are predicted by the CVAE, but a more reliable value is obtained by re-computing everything (descriptors, targets, and scores) from the compositions.
- preprocess_adaptative.py to prepare the input data
- model_cvae_adaptative.py implements the adaptive CVAE architecture with a custom multi-objective loss layer. The model combines standard VAE components (encoder, decoder, reparameterization trick) with a sophisticated loss function that balances reconstruction fidelity (MSE), latent space regularization (KL divergence), composition diversity (entropy penalty), and target property optimization (adaptive reward). The reward mechanism activates conditionally based on reconstruction and KL quality thresholds, enabling guided exploration of the composition space while maintaining model stability.
- train_adaptative.py orchestrates the end-to-end training pipeline with dynamic hyperparameter adaptation. Implements multiple callbacks that monitor training metrics and adjust β (KL weight), γ (entropy weight), and αᵢ (property reward weights) throughout training. Integrates active learning by generating candidate compositions during training, filtering them based on uncertainty, entropy, and property thresholds, and updating the reward strategy based on validation outcomes. This creates a closed-loop system where the model simultaneously learns to reconstruct data while optimizing for target material properties.

When the run of evaluate_adaptative.py is over - or if the user decides to stop it - and that new candidates were successfully generated, it is interesting to compute, from their composition, their descriptors + physical quantities + scores with C-HEA\1-Descriptors_for_HEA.py, and add this data as additional lines in the learning database for a future CVAE run.

