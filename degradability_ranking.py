import sys, os
import openpyxl
import itertools
import random
import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Draw import IPythonConsole
from sklearn import svm,tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))+'/Data')
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))+'/Model')

#pairwise calculation of a dataset
def transform_pairwise(X, y):
    X_new = []
    y_new = []
    xx = []
    yy = []
    valid_index = []
    comb = itertools.permutations(range(X.shape[0]), 2)
    if y is None:
        y_new = np.full(len(X), -1)
        comb = itertools.permutations(range(X.shape[0]), 2)
        for (i, j) in enumerate(comb):
            X_new.append(X[i] - X[j])
            valid_index.append([i,j])
        return np.asarray(X_new),np.asarray(y_new).ravel()

    else:
        y = np.asarray(y)
        if y.ndim == 1:
            y = np.c_[y, np.ones(y.shape[0])]
        comb = itertools.permutations(range(X.shape[0]), 2)
        for k, (i, j) in enumerate(comb):
            if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
                # skip if same target or different group
                continue
            X_new.append(X[i] - X[j])
            y_new.append(np.sign(y[i, 0] - y[j, 0]))
            xx.append([i,j,np.sign(y[i, 0] - y[j, 0])])
            # output balanced classes
            if y_new[-1] != (-1) ** k:
                y_new[-1] = - y_new[-1]
                X_new[-1] = - X_new[-1]
        x = np.asarray(X_new)
        y = np.asarray(y_new).ravel()
        return x, np.where(y < 0, -1,1)
    
# mol2vec representation
Zinc_model = word2vec.Word2Vec.load('Model/model_300dim.pkl')

def mol2vec(mols, model=Zinc_model):
    if type(mols) != list:
        x_sentence = [MolSentence(mol2alt_sentence(mols, 1))]
    else:
        x_sentence = [MolSentence(mol2alt_sentence(x, 1)) for x in mols]
    x_molvec=[DfVec(x) for x in sentences2vec(x_sentence, model, unseen='UNK')]
    x_molvec = np.array([x.vec for x in x_molvec])
    return x_molvec

# scaling
def autoscale(y):
    return (y - y.min()) / (y.max()-y.min())

# hyper parameter tuning with 5-folds cross validation
c_list = [1e-9,1e-7,1e-5,1e-4,1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2,2e-2,3e-2,5e-2,1e-1,1,10,50,1e2,1e3,1e5,1e10]
kf = KFold(n_splits=5,shuffle=True, random_state=9527)

def svm_cv(model,X,Y):
    cv_score = list()
    for train_index, test_index in kf.split(X, Y):
        x_train = X[train_index]
        y_train = Y[train_index]
        x_test = X[test_index]
        y_test = Y[test_index]
        model.fit(x_train, y_train)
        y_pred=model.predict(x_test)
        cv_score.append(accuracy_score(y_pred,y_test))
    return mean(cv_score)

# load datasets
df_lit = pd.read_excel('Data/literature.xlsx')
df_exp1 = pd.read_excel('Data/exp1.xlsx')
df_exp2 = pd.read_excel('Data/exp2.xlsx')

mols_lit = [Chem.MolFromSmiles(i) for i in df_lit['SMILES']]
deg_lit = df_lit['Degradability(Literature)']

mols_exp1 = [Chem.MolFromSmiles(i) for i in df_exp1['SMILES']]
deg_exp1 = df_exp1['Degradability(Exp)']

mols_exp2 = [Chem.MolFromSmiles(i) for i in df_exp2['SMILES']]
deg_exp2 = df_exp2['Degradability(Exp)']

# pairwise caculation of datasets separately and combine them together
x_labeled_lit, y_labeled_lit = transform_pairwise(mol2vec(mols_lit), deg_lit)
x_labeled_exp1, y_labeled_exp1 = transform_pairwise(mol2vec(mols_exp1), deg_exp1)
x_labeled_exp2, y_labeled_exp2 = transform_pairwise(mol2vec(mols_exp2), deg_exp2)

x_labeled = np.concatenate([x_labeled_lit, x_labeled_exp1, x_labeled_exp2])
y_labeled = np.concatenate([y_labeled_lit, y_labeled_exp1 ,y_labeled_exp2])



# Grid search-cv
# for i in c_list:
#     model_svc = svm.LinearSVC(C=i, dual=False, max_iter=10000)
#     print('C:',i , 'cv_acc', svm_cv(model_svc, x_labeled, y_labeled))

# C: 1e-09 cv_acc 0.6464162194394752
# C: 1e-07 cv_acc 0.6479785330948121
# C: 1e-05 cv_acc 0.675742397137746
# C: 0.0001 cv_acc 0.7113536076326774
# C: 0.001 cv_acc 0.8087298747763864
# C: 0.002 cv_acc 0.8226118067978533
# C: 0.003 cv_acc 0.8333929636255217
# C: 0.004 cv_acc 0.8380322003577818
# C: 0.005 cv_acc 0.8473225998807394
# C: 0.006 cv_acc 0.8473225998807394
# C: 0.007 cv_acc 0.8519618366129994 <--
# C: 0.008 cv_acc 0.8504114490161002

model_degradability = svm.LinearSVC(C=0.007,dual=False, max_iter=10000)
model_degradability.fit(x_labeled, y_labeled)
x_all = mol2vec(mols_lit+mols_exp1+mols_exp2)
y_pred =np.dot(x_all, model_degradability.coef_.ravel())
y_pred_s = autoscale(y_pred)
names = df_lit['Name'].tolist()+df_exp1['Name'].tolist()+df_exp2['Name'].tolist()
df_rank = pd.DataFrame({'name':names,'score':autoscale(y_pred_s)})
df_rank_res = df_rank.sort_values('score')
unified_ranking = dict(zip(df_rank_res['name'], df_rank_res['score']))
print(unified_ranking)

#{'Poly(isopropyl acrylate)': 0.0, 'Poly(isodecyl acrylate)': 0.12454962587763241, 'Poly(benzyl acrylate)': 0.24813249641515406, 'PET': 0.30315135478718386, 'PLLA': 0.3128833242372867, 'PLA': 0.3128833368415171, 'PS': 0.3846617462015356, 'Poly(2-methoxyethyl acrylate)': 0.39909693637417804, 'PP': 0.40131393952524397, 'Polyacetal': 0.46000736500351574, 'Poly(2-butoxyethyl acrylate)': 0.4600530398212392, 'PU': 0.5073934793929744, 'PCL': 0.5322939622721921, 'Poly(tetrafluoroethylene)': 0.5446108751331157, 'PC': 0.5867365084697796, 'Poly(hexamethylene sebacate)': 0.593622695858069, 'Poly(vinyl butyral)': 0.5979034610778075, 'Nylon6': 0.6419686158173797, 'Nylon66': 0.6503862490243772, 'PMMA': 0.6676282810654136, 'Nylon4': 0.6866152721562861, 'PEN': 0.7159537042700553, 'P3HB': 0.7233664982596433, 'PBSeb': 0.76362362655933, 'PPSeb': 0.7859468734615163, 'PBAZ': 0.7859468928489013, 'PPAz': 0.8082701485884867, 'PESeb': 0.8082701513437246, 'PPS': 0.8305934530633539, 'PEAz': 0.8305934617581082, 'PPPIM': 0.8529167768108115, 'PBAdip': 0.8529167879899133, 'PVA': 0.8647673309201457, 'PPAd': 0.8752401162253135, 'PPGI': 0.8975634407113782, 'PBS': 0.8975634781456812, 'Polychloroprene': 1.0}

# DT analysis of Ranking result using molecular descriptors
model_DTR = DecisionTreeRegressor(random_state=445,
                                  max_leaf_nodes=10,
                                  max_depth=5,
                                  splitter='random')
model_DTR.fit(x_dt,y_dt)

fn=descriptors.columns
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize =(15,7), dpi=300)
tree.plot_tree(model_DTR,
               feature_names = fn, 
               filled = True)
plt.show()
fig.savefig('tree_vis.png')
