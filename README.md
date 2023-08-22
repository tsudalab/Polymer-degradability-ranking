# Polymer-degradability-ranking
### Revealing Factors Influencing Polymer Degradation with Rank-based Machine Learning
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8268022.svg)](https://doi.org/10.5281/zenodo.8268022)         

![](https://github.com/tsudalab/Polymer-degradability-ranking/blob/main/ranking_result.png)   

## Data  
The experimental data of polymer degradability and literature data can be found in /Data  

The predicted degradability of PolyInfo data in model applicability domain can be found in (https://github.com/tsudalab/Polymer-degradability-ranking/blob/main/degradabilty_result_of_polyinfo.csv). The detailed information about polymers can be tracked through PID in (https://polymer.nims.go.jp/)  

## Model  
The model for degradability ranking can be found in /Model. 

## Requirement
** Python  
** Mol2vec (https://github.com/samoturk/mol2vec)  
** Gensim (https://radimrehurek.com/gensim/)  
** Scikit-learn 
** RDKit

## Usage

### Loading Datasets  
The script is designed to work with following Excel files containing molecular information:

    'Data/literature.xlsx'
    'Data/exp1.xlsx'
    'Data/exp2.xlsx'

### Pairwise Calculation  
First, the `mol2vec` calculation is applied to the molecules to obtain embemdding vectors. Then, pairwise calculation of datasets can be performed separately and then combined together. Example:

```python
x_labeled_lit, y_labeled_lit = transform_pairwise(mol2vec(mols_lit), deg_lit)
x_labeled_exp1, y_labeled_exp1 = transform_pairwise(mol2vec(mols_exp1), deg_exp1)
x_labeled_exp2, y_labeled_exp2 = transform_pairwise(mol2vec(mols_exp2), deg_exp2)

x_labeled = np.concatenate([x_labeled_lit, x_labeled_exp1, x_labeled_exp2])
y_labeled = np.concatenate([y_labeled_lit, y_labeled_exp1 ,y_labeled_exp2])
```
### Model Training and Prediction  
SVM is used to train the degradability model. The hyperparameter is optimized using grid search. The trained ranking model then applied to create unified ranking

### Decision Tree Analysis  
Decision tree analysis of the ranking result using molecular descriptors is provided at the end of the script.

## How to Run  
Execute the script by running
```bash
python degradability_ranking.py
```

## Applicability Domain determination of Polyinfo data  
The code for Applicability Domain determination can be found in (https://github.com/onecoinbuybus/KNN-Applicability-Domain/tree/main) 

