
# materials_inverse_design_pipeline
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17567098.svg)](https://doi.org/10.5281/zenodo.17567098)

This repository contains the code and configuration files for the project *Materials Inverse Design Pipeline*,
developed in the context of research and publication for **npj Computational Materials**.

## Project overview
This pipeline provides a fully reproducible framework for data preparation, model training, evaluation, and
visualization in the context of inverse design of materials. It is intended to accompany the submitted article
and ensure reproducibility of the computational results.

## Structure
```
materials_inverse_design_pipeline/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ environment.yml
├─ requirements.txt
├─ CITATION.cff
├─ configs/
│  ├─ base.yaml
├─ data/
│  └─ sample/
└─ results/
```

## Quick start

### 1. Create the Conda environment
```bash
conda env create -f environment.yml
conda activate materials-env
```

### 2. Run the pipeline 
Execute scripts following the alphabetical order of the folders and the numerical order of the scripts inside. 
Read the readme file in every folder before running scripts.

### 3. Results
Each folder readme file specifies where generated outputs (metrics, figures, tables) are stored; the most significant were copied under the `results/` directory.

## Data
Only a minimal sample dataset is provided (`data/sample/`): the initial raw data to initiate the pipeline, and some of the most advanced results.  
Full datasets are archived separately (see *Data availability* in the paper / thesis).

## License
Distributed under the MIT License. See `LICENSE` for details.

## Citation
Please cite this repository using the metadata in `CITATION.cff`.
