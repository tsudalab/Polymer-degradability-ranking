# Polymer-degradability-ranking
### Revealing Factors Influencing Polymer Degradation with Rank-based Machine Learning
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8268022.svg)](https://doi.org/10.5281/zenodo.8268022)         

![](https://github.com/tsudalab/Polymer-degradability-ranking/blob/main/ranking_result.png)   

## Data  
The experimental data of polymer degradability and literature data can be found in /Data  

The predicted degradability of PolyInfo data in model applicability domain can be found in (https://github.com/tsudalab/Polymer-degradability-ranking/blob/main/degradabilty_result_of_polyinfo.csv). The detailed information about polymers can be tracked through PID in (https://polymer.nims.go.jp/)  

## Model  
The model for degradability ranking can be found in /Model. 

## Usage
## Usage

### Loading Datasets
The script is designed to work with specific Excel files containing molecular information. Example files are expected in this format:

    'Data/literature.xlsx'
    'Data/exp1.xlsx'
    'Data/exp2.xlsx'

### Pairwise Calculation

Pairwise calculation of datasets can be performed separately and then combined together. Example:

```python
x_labeled_lit, y_labeled_lit = transform_pairwise(mol2vec(mols_lit), deg_lit)
```



## How to Run  
Execute the script by running
```bash
python degradability_ranking.py
```

## Applicability Domain determination of Polyinfo data  
The code for Applicability Domain determination can be found in (https://github.com/onecoinbuybus/KNN-Applicability-Domain/tree/main) 

