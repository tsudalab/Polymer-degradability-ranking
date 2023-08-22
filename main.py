import sys, os
import openpyxl
import itertools
import random
import pickle 
import pandas as pd
import numpy as np
from statistics import mean
from rdkit import Chem
from sklearn import svm,tree
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))+'/Data')
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))+'/Model')

Zinc_model = word2vec.Word2Vec.load("Model/model_300dim.pkl")


class degradability_ranking:
    def __init__(self):
        self.Zinc_model = word2vec.Word2Vec.load("Model/model_300dim.pkl")
        self.kf = KFold(n_splits=5, shuffle=True, random_state=9527)
        self.model_degradability = svm.LinearSVC(C=0.007, dual=False, max_iter=10000)
        self.model_degradability = None

    def transform_pairwise(self, X, y):
        X_new = []
        y_new = []
        xx = []
        yy = []
        valid_index = []
        comb = itertools.permutations(range(X.shape[0]), 2)
        if y is None:
            y_new = np.full(len(X), -1)
            comb = itertools.permutations(range(X.shape[0]), 2)
            for i, j in enumerate(comb):
                X_new.append(X[i] - X[j])
                valid_index.append([i, j])
            return np.asarray(X_new), np.asarray(y_new).ravel()

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
                xx.append([i, j, np.sign(y[i, 0] - y[j, 0])])
                # output balanced classes
                if y_new[-1] != (-1) ** k:
                    y_new[-1] = -y_new[-1]
                    X_new[-1] = -X_new[-1]
            x = np.asarray(X_new)
            y = np.asarray(y_new).ravel()
            return x, np.where(y < 0, -1, 1)

    def mol2vec(self, mols, model=Zinc_model):
        if type(mols) != list:
            x_sentence = [MolSentence(mol2alt_sentence(mols, 1))]
        else:
            x_sentence = [MolSentence(mol2alt_sentence(x, 1)) for x in mols]
        x_molvec = [DfVec(x) for x in sentences2vec(x_sentence, model, unseen="UNK")]
        x_molvec = np.array([x.vec for x in x_molvec])
        return x_molvec

    def autoscale(self, y):
        return (y - y.min()) / (y.max() - y.min())

    def svm_cv(self, model, X, Y):
        cv_score = list()
        for train_index, test_index in kf.split(X, Y):
            x_train = X[train_index]
            y_train = Y[train_index]
            x_test = X[test_index]
            y_test = Y[test_index]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            cv_score.append(accuracy_score(y_pred, y_test))
        return mean(cv_score)

    def train(self, train_data_list=None):
        print("Starting the training process...")
        print("Loading datasets...")
        df_lit = pd.read_excel("Data/literature.xlsx")
        df_exp1 = pd.read_excel("Data/exp1.xlsx")
        df_exp2 = pd.read_excel("Data/exp2.xlsx")

        mols_lit = [Chem.MolFromSmiles(i) for i in df_lit["SMILES"]]
        deg_lit = df_lit["Degradability"]

        mols_exp1 = [Chem.MolFromSmiles(i) for i in df_exp1["SMILES"]]
        deg_exp1 = df_exp1["Degradability"]

        mols_exp2 = [Chem.MolFromSmiles(i) for i in df_exp2["SMILES"]]
        deg_exp2 = df_exp2["Degradability"]

        # pairwise caculation of datasets separately and combine them together
        print("Performing pairwise calculations...")
        x_labeled_lit, y_labeled_lit = self.transform_pairwise(self.mol2vec(mols_lit), deg_lit)
        x_labeled_exp1, y_labeled_exp1 = self.transform_pairwise(
            self.mol2vec(mols_exp1), deg_exp1
        )
        x_labeled_exp2, y_labeled_exp2 = self.transform_pairwise(
            self.mol2vec(mols_exp2), deg_exp2
        )

        x_labeled = np.concatenate([x_labeled_lit, x_labeled_exp1, x_labeled_exp2])
        y_labeled = np.concatenate([y_labeled_lit, y_labeled_exp1, y_labeled_exp2])

        if train_data_list is not None:
            print("Processing additional training data...")
            if isinstance(train_data_list, str):
                train_data_list = [train_data_list]

            for file_name in train_data_list:
                df_new = pd.read_excel(f"Data/{file_name}.xlsx")
                mols_new = [Chem.MolFromSmiles(i) for i in df_new["SMILES"]]
                deg_new = df_new["Degradability"]
                x_new, y_new = self.transform_pairwise(self.mol2vec(mols_new), deg_new)
                x_labeled = np.concatenate([x_labeled, x_new])
                y_labeled = np.concatenate([y_labeled, y_new])

            c_list = [
                1e-9,
                1e-7,
                1e-5,
                1e-4,
                1e-3,
                2e-3,
                3e-3,
                4e-3,
                5e-3,
                6e-3,
                7e-3,
                8e-3,
                9e-3,
                1e-2,
                2e-2,
                3e-2,
                5e-2,
                1e-1,
                1,
                10,
                50,
                1e2,
                1e3,
                1e5,
                1e10,
            ]

            def get_best_C(x_labeled, y_labeled, c_list, cv_folds=5):
                def svm_cv(model, X, y):
                    return cross_val_score(model, X, y, cv=cv_folds).mean()

                best_c = None
                best_acc = 0
                previous_acc = 0

                for i in c_list:
                    model_svc = svm.LinearSVC(C=i, dual=False, max_iter=10000)
                    cv_acc = svm_cv(model_svc, x_labeled, y_labeled)

                    if cv_acc > best_acc:
                        best_acc = cv_acc
                        best_c = i

                    if cv_acc < previous_acc:
                        break

                    previous_acc = cv_acc

                return best_c

            best_c = get_best_C(x_labeled, y_labeled, c_list, cv_folds=5)
            print(f"Best C value found: {best_c}")
            model_degradability = svm.LinearSVC(C=best_c, dual=False, max_iter=10000)
            with open("Model/update_model.pickle", "wb") as f:
                pickle.dump(model_degradability, f)
            print("Training process completed with additional data.")
            return model_degradability

        else:
            print("Training process completed without additional data.")
            model_degradability = svm.LinearSVC(C=0.007, dual=False, max_iter=10000)
            model_degradability.fit(x_labeled, y_labeled)
            return model_degradability

    def predict(self, smiles, model="deg_model.pickle", track="sp"):
        model_path = f"Model/{model}"
        print(f"Loading model from {model_path}...") # 输出正在加载的模型路径
        with open(model_path, "rb") as f:
            ranking_model = pickle.load(f)
        print("Model loaded.")
        if isinstance(smiles, str):
            smiles = [smiles]  
        print(f"Predicting for {len(smiles)} molecules...")
        mols = [Chem.MolFromSmiles(i) for i in smiles]
        X_pred = self.mol2vec(mols)
        result_dict = {"ranking": []}
        if track == "sp":
            y_pred = np.dot(X_pred, ranking_model.coef_.ravel())
            y_pred_s = self.autoscale(y_pred)

            sorted_results = sorted(zip(smiles, y_pred_s), key=lambda x: x[1])

            result_dict["ranking"] = []
            for rank, (smile, score) in enumerate(sorted_results, 1):
                result_dict["ranking"].append(
                    {"smiles": smile, "rank": rank, "score": score}
                )

        elif track == "c":
            df_lit = pd.read_excel("Data/literature.xlsx")
            df_exp1 = pd.read_excel("Data/exp1.xlsx")
            df_exp2 = pd.read_excel("Data/exp2.xlsx")

            mols_lit = [Chem.MolFromSmiles(i) for i in df_lit["SMILES"]]
            mols_exp1 = [Chem.MolFromSmiles(i) for i in df_exp1["SMILES"]]
            mols_exp2 = [Chem.MolFromSmiles(i) for i in df_exp2["SMILES"]]

            x_all = np.concatenate(
                [self.mol2vec(mols_lit), self.mol2vec(mols_exp1), self.mol2vec(mols_exp2), X_pred]
            )
            y_pred = np.dot(x_all, ranking_model.coef_.ravel())
            y_pred_s = self.autoscale(y_pred)


            results = list(zip(df_lit["SMILES"].tolist() + df_exp1["SMILES"].tolist() + df_exp2["SMILES"].tolist() + smiles, y_pred_s))
            sorted_results = sorted(results, key=lambda x: x[1])

            result_dict = {"ranking": []}
            for rank, (smile, score) in enumerate(sorted_results, 1):
                if smile in smiles:
                    nearby_smiles = {}

                    if rank > 1:
                        nearby_smiles["previous"] = sorted_results[rank - 2]

            
                    nearby_smiles["current"] = {"smiles": smile, "rank": rank, "score": score}

                    if rank < len(sorted_results):
                        nearby_smiles["next"] = sorted_results[rank]

                    result_dict["ranking"].append(nearby_smiles)

            print("Ranking within the complete dataset and nearby molecules:")

        print(result_dict)
        return result_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or Predict Degradability Ranking")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Sub-command for training
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("files", nargs="*", help="List of Excel files for training")

    # Sub-command for predicting
    predict_parser = subparsers.add_parser("predict", help="Predict using the model")
    predict_parser.add_argument("smiles", nargs="+", help="List of SMILES strings")
    predict_parser.add_argument(
        "-c", help="Compare with default data", action="store_true"
    )
    predict_parser.add_argument("-sp", help="Default prediction", action="store_true")
    predict_parser.add_argument("--model", help="Specify model file", default="deg_model.pickle")

    args = parser.parse_args()

    ranking_model = degradability_ranking()

    if args.command == "train":
        if args.files:
            files = [file for file in args.files]
            ranking_model.train(train_data_list=files)

    elif args.command == "predict":
        track = "sp" if args.sp else "c" if args.c else "sp"
        if track == "sp" and len(args.smiles) < 2:
            print("Error: Must provide at least two SMILES strings when using 'sp' tracking.")
            sys.exit(1)
        ranking_model.predict(args.smiles, model=args.model, track=track)


    # python main.py train exp1 exp2...
    # python main.py predict '*C*' '*CC*' -sp
    # python main.py predict '*C*' -c 
