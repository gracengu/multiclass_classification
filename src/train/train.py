import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pickle

import missingno as msno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

import sklearn
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, auc, \
    classification_report, multilabel_confusion_matrix, precision_recall_curve, roc_curve, average_precision_score
from lightgbm import LGBMClassifier

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier

from src.config import Config
from src.analysis import Analysis

pd.set_option('display.max_rows', 500)

class Train(Config):

    def __init__(self):
        self.data = {}

    def dataset_prep(self, retrain_pca=False, save_pca=False):

        # Set 1: Raw Data + Missing Data Deletion
        data = analysis.read_file()
        raw_data = data[~data.isna().any(axis=1)].reset_index(drop=True)
        dataset1 = raw_data.copy()
        missing_data = pd.DataFrame(raw_data.isna().sum()[raw_data.isna().sum() != 0], columns=["missing_count"])
        missing_data["percentage"] = missing_data.missing_count*100/raw_data.shape[0]

        # Set 2: Raw Data + Missing Data Deletion + Outlier Removal (Set 1 + Outlier Removal)
        # Outlier Removal of Normal Class where Minority Class have same behavior
        # Selected columns are also columns used for flagging (Feature Engineering)
        selected_columns = Config.ANALYSIS_CONFIG["XCHART_COLUMNS"]
        numerical_columns = raw_data[selected_columns].select_dtypes(["float", "int"])
        clean_df = analysis.outlier_identification(raw_data, numerical_columns, mode="outlier_removal")
        # Since the inherent missingess is from majority class and the maximum percentage missingness is 5%, \
        # the rows are simply removed
        missing_data = pd.DataFrame(clean_df.isna().sum()[clean_df.isna().sum() != 0], columns=["missing_count"])
        missing_data["percentage"] = missing_data.missing_count*100/clean_df.shape[0]
        dataset2 = clean_df[~clean_df.isna().any(axis=1)].reset_index(drop=True)

        # Set 3: Raw Data + Missing Data Deletion + Outlier Removal + Feature Transformation (Set 2 + Feature Transformation)
        transformed_dict, _ = analysis.data_transformation(dataset2)
        transformed_df = pd.concat([transformed_dict["SYMMETRICAL_DATA"],
                                    transformed_dict["MODPOS_TRANSFORMED"],
                                    transformed_dict["MODNEG_TRANSFORMED"],
                                    transformed_dict["HIGHPOS_TRANSFORMED"],
                                    transformed_dict["HIGHNEG_TRANSFORMED"],
                                    dataset2[["105", "147"]], # Include bimodal distributions for now
                                    dataset2.select_dtypes("category")], axis=1)
        dataset3 = transformed_df.copy()


        # Set 4: Raw Data + Missing Data Deletion + Outlier Removal + Feature Transformation + Feature Engineering (Flag)
        selected_columns = Config.ANALYSIS_CONFIG["TRANSFORMED_COLUMNS"]
        numerical_columns = dataset3[selected_columns].select_dtypes(["float", "int"])
        flag_data = analysis.outlier_identification(dataset3, numerical_columns, mode='feature_engineering')
        dataset4 = pd.concat([flag_data.reset_index(drop=True),
                            dataset3[selected_columns].select_dtypes(["category"])], axis=1)


        # Save dataset
        dataset1.to_csv("./data/dataset1.csv", index=False)
        dataset2.to_csv("./data/dataset2.csv", index=False)
        dataset3.to_csv("./data/dataset3.csv", index=False)
        dataset4.to_csv("./data/dataset4.csv", index=False)

        # Read dataset and change datatype
        dataset1 = analysis.read_file("./data/dataset1.csv")
        dataset2 = analysis.read_file("./data/dataset2.csv")
        dataset3 = analysis.read_file("./data/dataset3.csv")
        dataset4 = analysis.read_file("./data/dataset4.csv")

        # Set 5: Set 4 -> Pure PCA + Target
        pca_df1 = analysis.pca_transformation(dataset1, retrain=retrain_pca, fname="./models/pca_v3.sav", save=save_pca)
        pca_df2 = analysis.pca_transformation(dataset2, retrain=retrain_pca, fname="./models/pca_v4.sav", save=save_pca)
        pca_df3 = analysis.pca_transformation(dataset3, retrain=retrain_pca, fname="./models/pca_v3.sav", save=save_pca)
        pca_df4 = analysis.pca_transformation(dataset4, retrain=retrain_pca, fname="./models/pca_v4.sav", save=save_pca)
        pca_df1.to_csv("./data/pca_dataset1.csv", index=False)
        pca_df2.to_csv("./data/pca_dataset2.csv", index=False)
        pca_df3.to_csv("./data/pca_dataset3.csv", index=False)
        pca_df4.to_csv("./data/pca_dataset4.csv", index=False)
        pca_df1 = analysis.read_file("./data/pca_dataset1.csv")
        pca_df2 = analysis.read_file("./data/pca_dataset2.csv")
        pca_df3 = analysis.read_file("./data/pca_dataset3.csv")
        pca_df4 = analysis.read_file("./data/pca_dataset4.csv")

        # Set 6: Hybrid of all (Transformed, Engineering)
        combined_df1 = pd.concat([dataset1.loc[:, ~dataset1.columns.isin(["target"])].reset_index(drop=True),
                                pca_df1.reset_index(drop=True)], axis=1)
        combined_df2 = pd.concat([dataset2.loc[:, ~dataset2.columns.isin(["target"])].reset_index(drop=True),
                                pca_df2.reset_index(drop=True)], axis=1)
        combined_df3 = pd.concat([dataset3.loc[:, ~dataset3.columns.isin(["target"])].reset_index(drop=True),
                                pca_df3.reset_index(drop=True)], axis=1)
        combined_df4 = pd.concat([dataset4.loc[:, ~dataset4.columns.isin(["target"])].reset_index(drop=True),
                                pca_df4.reset_index(drop=True)], axis=1)

        combined_df1.to_csv("./data/combined_dataset1.csv", index=False)
        combined_df2.to_csv("./data/combined_dataset2.csv", index=False)
        combined_df3.to_csv("./data/combined_dataset3.csv", index=False)
        combined_df4.to_csv("./data/combined_dataset4.csv", index=False)


    def feature_selection(self, X_train, y_train, fname, retrain=False, num_cols="all", threshold=.5):

        random.seed(123)
        # Numerical input Categorical Output: ANOVA 
        X_anova = X_train.select_dtypes(["float", "int"])
        filename = "./models/models_new/anova_{}".format(fname)
        if retrain:
            fs = SelectKBest(score_func=f_classif, k=num_cols).fit(X_anova, y_train)
            pickle.dump(fs, open(filename, 'wb'))
        fs = pickle.load(open(filename, 'rb'))
        X_selected = fs.transform(X_anova)
        anova_df = pd.DataFrame({"features_selected" : list(X_anova.loc[:, fs.get_support()].columns),
                                "features_pvalues": list(fs.pvalues_[fs.get_support()])})
        
        anova_df = anova_df.loc[anova_df.features_pvalues <= threshold, :]
        anova_df = anova_df.sort_values(by="features_pvalues", ascending=True)

        # Numerical input Categorical Output: Chi2 
        X_chi2 = X_train.select_dtypes(["category"])
        filename = "./models/models_new/chi2_{}".format(fname)
        if retrain:
            fs = SelectKBest(score_func=chi2, k=num_cols).fit(X_chi2, y_train)
            pickle.dump(fs, open(filename, 'wb'))
        fs = pickle.load(open(filename, 'rb'))
        X_selected = fs.transform(X_chi2)
        chi2_df = pd.DataFrame({"features_selected" : list(X_chi2.loc[:, fs.get_support()].columns),
                                "features_pvalues": list(fs.pvalues_[fs.get_support()])})
        chi2_df = chi2_df.loc[chi2_df.features_pvalues <= threshold, :]
        chi2_df = chi2_df.sort_values(by="features_pvalues", ascending=False)

        return anova_df, chi2_df

    def feature_importance(self, data, title, fontsize=20):
    
        fig = plt.figure(figsize=(20,50))
        plt.barh(data["features_selected"], data["features_pvalues"])
        plt.title(title, fontsize=fontsize)
        plt.xlabel("features_pvalues", fontsize=fontsize)
        plt.ylabel("features", fontsize=fontsize)

        return fig

    def oversampling(self, X_train, y_train, plot=False):
    
        oversample = SMOTE(random_state=123)
        X_otrain,y_otrain = oversample.fit_resample(X_train,y_train)
        if plot:
            display(y_otrain.value_counts(normalize=True).plot.pie())
            
        return X_otrain, y_otrain

    def print_confusion_matrix(self, confusion_matrix, axes, class_label, class_names, fontsize=14):

        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)

        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axes.set_ylabel('True label')
        axes.set_xlabel('Predicted label')
        axes.set_title("Confusion Matrix for the class - " + class_label)
        return fig
        
    def cm_single(self, y_test, y_pred, fontsize=14):
        
        labels = ["".join("c" + str(i[0])) for i in pd.DataFrame(y_test).value_counts().index]
        df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index=labels, columns=labels)
        fig = plt.figure(figsize=(6,4))
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False)
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        heatmap.set_ylabel('True label')
        heatmap.set_xlabel('Predicted label')
        heatmap.set_title("Confusion Matrix for Binary Label" )
        return fig
            

    def cm_plot(self, y_test, y_pred, nrow=3, ncol=2):
        
        labels = ["".join("c" + str(i[0])) for i in pd.DataFrame(y_test).value_counts().index]
        cm = multilabel_confusion_matrix(y_test, y_pred)
        if nrow == 1: 
            figsize = (8,4)
        else:
            figsize = (12,7)
        fig, ax = plt.subplots(nrow, ncol, figsize=figsize)
        for axes, cfs_matrix, label in zip(ax.flatten(), cm, labels):
            print_confusion_matrix(cfs_matrix, axes, label, ["0", "1"])
        if nrow == 3:
            fig.delaxes(ax[2,1])
        fig.tight_layout()
        return fig
        
    def pr_auc(self, X_test, y_test, n_classes):
        
        fig, ax = plt.subplots(2, 1, figsize=(12, 7))
        
        # Precision-Recall Curve
        y_score = model.predict_proba(X_test)
        y_test = label_binarize(y_test, classes=[*range(n_classes)])
        precision = dict()
        recall = dict()
        
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],y_score[:, i])
            ax[0].plot(recall[i], precision[i], lw=2, label='class {}'.format(i))
        no_skill = len(y[y==1]) / len(y)
        ax[0].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        ax[0].set_xlabel("recall")
        ax[0].set_ylabel("precision")
        ax[0].legend(loc="best")
        ax[0].set_title("precision vs. recall curve")

        # ROC curve
        fpr = dict()
        tpr = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            ax[1].plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))
        ax[1].plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        ax[1].set_xlabel("false positive rate")
        ax[1].set_ylabel("true positive rate")
        ax[1].legend(loc="best")
        ax[1].set_title("ROC curve")
        
        fig.tight_layout()
        return fig

    def traintest_split(self, data, test_size=0.3):

        X, y = data.loc[:,~data.columns.isin(["target"])], data.loc[:,data.columns.isin(["target"])]
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size)

        return X_train, X_test, y_train, y_test
        
    def class_count(self, y_test, y_pred):

        true = pd.DataFrame(y_test.value_counts())
        pred = pd.DataFrame(list(np.unique(y_pred, return_counts=True)[1]))
        pred.index = list(np.unique(y_pred, return_counts=True)[0])
        final = pd.concat([true, pred], axis=1).rename(columns={0:"pred"})
        
        return final

    def plot_graphs(self, X_test, y_test, y_pred, model, fontsize=16):

        algo_name = type(model).__name__
        y_proba = model.predict_proba(X_test)
        n_classes = len(y_test["target"].unique())
        onehotencoder = OneHotEncoder()
        y_enc = onehotencoder.fit_transform(np.array(y_test).reshape(-1,1)).toarray()
        
        fig, (ax1, ax2, ax3)  = plt.subplots(1, 3, figsize=(12, 5))
        plt.style.use('default')
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_enc[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            ax1.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel("false positive rate")
        ax1.set_ylabel("true positive rate")
        ax1.legend(loc="lower right", prop={'size': 10})
        ax1.set_title('ROC to multi-class')
        fig.suptitle(algo_name, fontsize=16)
        
        plt.style.use('default')
        precision = dict()
        recall = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_enc[:, i], y_proba[:, i])
            ax2.plot(recall[i], precision[i], lw=2, label='PR Curve of class {}'.format(i))
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel("recall")
        ax2.set_ylabel("precision")
        ax2.legend(loc="lower right", prop={'size': 10})
        ax2.set_title('Precision-Recall to multi-class')
        fig.suptitle(algo_name, fontsize=16)
        
        labels = ["".join("c" + str(i[0])) for i in pd.DataFrame(y_test).value_counts().index]
        df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), index=labels, columns=labels)
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=ax3)
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        heatmap.set_ylabel('True label')
        heatmap.set_xlabel('Predicted label')
        heatmap.set_title("Confusion Matrix for Binary Label" )
        plt.suptitle(algo_name, fontsize=16)
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        display(fig)

    def data_prep(self, data, fname, retrain=False, test_size=0.3):

        X_train, X_test, y_train, y_test = self.traintest_split(data, test_size=test_size)
        ANOVA_fs, chi2_fs = self.feature_selection(X_train, y_train, fname, retrain=retrain, num_cols="all")
        feature_list = list(ANOVA_fs["features_selected"].values) + list(chi2_fs["features_selected"].values) + ["target"]
        temp = data.loc[:, data.columns.isin(feature_list)]
        cols = ["12", "64", "95"] 
        temp[cols] = temp[cols].astype("int64")
        X_train, X_test, y_train, y_test = self.traintest_split(data, test_size=test_size)
        X_otrain, y_otrain = self.oversampling(X_train, y_train, plot=True)

        return X_train, X_test, y_train, y_test, X_otrain, y_otrain

    def baseline_model(self, data, retrain=False):

        X_train, X_test, y_train, y_test, X_otrain, y_otrain = self.data_prep(data, 'fs1.sav', retrain=retrain)

        # Logistic Regression
        filename = './models/models_new/logreg_baseline_dataset1.sav'
        if retrain:
            lr_dataset1 = LogisticRegression(random_state=123)
            lr_dataset1.fit(X_train, y_train)
            pickle.dump(lr_dataset1, open(filename, 'wb'))
        lr_dataset1 = pickle.load(open(filename, 'rb'))
        y_pred = lr_dataset1.predict(X_test)
        print("Evaluation metrics for Logistic Regression:")
        report_logreg = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_logreg)
        self.plot_graphs(X_test, y_test, y_pred, lr_dataset1)

        # Weighted Logistic Regression
        filename = './models/models_new/weightedlogreg_baseline_dataset1.sav'
        if retrain:
            lr_dataset2 = LogisticRegression(class_weight='balanced', random_state=123)
            lr_dataset2.fit(X_train, y_train)
            pickle.dump(lr_dataset2, open(filename, 'wb'))
        lr_dataset2 = pickle.load(open(filename, 'rb'))
        y_pred = lr_dataset2.predict(X_test)
        print("Evaluation metrics for Weighted Logistic Regression:")
        report_weightlogreg = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_weightlogreg)
        self.plot_graphs(X_test, y_test, y_pred, lr_dataset2)

        # Random Forest Classifier
        filename = './models/models_new/rf_baseline_dataset1.sav'
        if retrain:
            rf_model = RandomForestClassifier(random_state=123).fit(X_train, y_train)
            pickle.dump(rf_model, open(filename, 'wb'))
        rf_model = pickle.load(open(filename, 'rb'))
        y_pred = rf_model.predict(X_test)
        print("Evaluation metrics for Random Forest Classifier:")
        report_rf = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_rf)
        self.plot_graphs(X_test, y_test, y_pred, rf_model)

        # LightGBM
        filename = './models/models_new/lgbm_baseline_dataset0.sav'
        if retrain:
            lgbm_model = LGBMClassifier(random_state=123).fit(X_train, y_train)
            pickle.dump(lgbm_model, open(filename, 'wb'))
        lgbm_model = pickle.load(open(filename, 'rb'))
        y_pred = lgbm_model.predict(X_test)
        print("Evaluation metrics for Light Gradient Boosting Classifier:")
        report_lgbm = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_lgbm)
        self.plot_graphs(X_test, y_test, y_pred, lgbm_model)

        # Linear SVC
        filename = './models/models_new/svc_baseline_dataset1.sav'
        if retrain:
            svm_model = SVC(kernel='linear',probability=True, random_state=123).fit(X_train, y_train)
            pickle.dump(svm_model, open(filename, 'wb'))
        svm_model = pickle.load(open(filename, 'rb'))
        y_pred = svm_model.predict(X_test)
        print("Evaluation metrics for Linear Support Vector Classifier:")
        report_svc = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_svc)
        self.plot_graphs(X_test, y_test, y_pred, svm_model)

        # RBF SVC
        filename = './models/models_new/svk_baseline_dataset1.sav'
        if retrain:
            svm_kernel = SVC(kernel='rbf',probability=True, random_state=123).fit(X_train, y_train)
            pickle.dump(svm_kernel, open(filename, 'wb'))
        svm_kernel = pickle.load(open(filename, 'rb'))
        y_pred = svm_model.predict(X_test)
        print("Evaluation metrics for Radial-Basis Support Vector Classifier:")
        report_svk = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_svk)
        self.plot_graphs(X_test, y_test, y_pred, svm_kernel)

        # return report_logreg, report_weightlogreg, report_rf,report_lgbm, report_svc, report_svk

    def model_bysubsets(self, retrain=False):
        
        analysis = Analysis()
        combined_df2 = analysis.read_file("./data/combined_dataset2.csv")
        combined_df3 = analysis.read_file("./data/combined_dataset3.csv")
        combined_df4 = analysis.read_file("./data/combined_dataset4.csv")

        X_train, X_test, y_train, y_test, X_otrain, y_otrain = self.data_prep(combined_df2, 'fs2.sav', retrain=True)
        filename = './models/models_new/lgbm_dataset2.sav'
        if retrain:
            lgbm_model = LGBMClassifier(random_state=123).fit(X_train, y_train)
            pickle.dump(lgbm_model, open(filename, 'wb'))
        lgbm_model = pickle.load(open(filename, 'rb'))
        y_pred = lgbm_model.predict(X_test)
        print("Evaluation metrics for LGBM with dataset 2:")
        report_subset2 = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_subset2)
        self.plot_graphs(X_test, y_test, y_pred, lgbm_model)

        X_train, X_test, y_train, y_test, X_otrain, y_otrain = self.data_prep(combined_df3, 'fs3.sav', retrain=True)
        filename = './models/models_new/lgbm_dataset3.sav'
        if retrain:
            lgbm_model = LGBMClassifier(random_state=123).fit(X_train, y_train)
            pickle.dump(lgbm_model, open(filename, 'wb'))
        lgbm_model = pickle.load(open(filename, 'rb'))
        y_pred = lgbm_model.predict(X_test)
        print("Evaluation metrics for LGBM with dataset 3:")
        report_subset3 = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_subset3)
        self.plot_graphs(X_test, y_test, y_pred, lgbm_model)

        X_train, X_test, y_train, y_test, X_otrain, y_otrain = self.data_prep(combined_df4, 'fs4.sav', retrain=True)
        filename = './models/models_new/lgbm_dataset4.sav'
        if retrain:
            lgbm_model = LGBMClassifier(random_state=123).fit(X_train, y_train)
            pickle.dump(lgbm_model, open(filename, 'wb'))
        lgbm_model = pickle.load(open(filename, 'rb'))
        y_pred = lgbm_model.predict(X_test)
        print("Evaluation metrics for LGBM with dataset 4:")
        report_subset4 = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_subset4)
        self.plot_graphs(X_test, y_test, y_pred, lgbm_model)

        # return report_subset2, report_subset3, report_subset4

    def model_tuning(self, data, retrain=False):

        X_train, X_test, y_train, y_test, X_otrain, y_otrain = self.data_prep(data, 'fs4.sav', retrain=retrain)
        filename = './models/models_new/lgbm_dataset4.sav'
        if retrain:
            lgbm_model = LGBMClassifier(random_state=123).fit(X_train, y_train)
            pickle.dump(lgbm_model, open(filename, 'wb'))
        lgbm_model = pickle.load(open(filename, 'rb'))
        y_pred = lgbm_model.predict(X_test)
        print("Evaluation metrics for LGBM without oversampling:")
        report_subset4 = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_subset4)
        self.plot_graphs(X_test, y_test, y_pred, lgbm_model)

        X_train, X_test, y_train, y_test, X_otrain, y_otrain = self.data_prep(data, 'fs4.sav', retrain=retrain)
        filename = './models/models_new/lgbm_dataset4_oversampling.sav'
        if retrain:
            lgbm_model = LGBMClassifier(random_state=123).fit(X_otrain, y_otrain)
            pickle.dump(lgbm_model, open(filename, 'wb'))
        lgbm_model = pickle.load(open(filename, 'rb'))
        y_pred = lgbm_model.predict(X_test)
        print("Evaluation metrics for LGBM with dataset 4 with oversampling:")
        report_oversampling = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_oversampling)
        self.plot_graphs(X_test, y_test, y_pred, lgbm_model)

        X_train, X_test, y_train, y_test, X_otrain, y_otrain = self.data_prep(data, 'fs4.sav', retrain=retrain)
        filename = './models/models_new/lgbm_dataset4_tuned.sav'
        if retrain:
            lgbm_model = LGBMClassifier(learning_rate=0.6960720013050441, max_depth= 6, n_estimators= 1181, num_leaves= 69, \
            subsample= 0.3646666061385309, random_state=123).fit(X_otrain, y_otrain)
            pickle.dump(lgbm_model, open(filename, 'wb'))
        lgbm_model = pickle.load(open(filename, 'rb'))
        y_pred = lgbm_model.predict(X_test)
        print("Evaluation metrics for LGBM with dataset 4 with hyperparameter tuning:")
        report_oversampling_tuned = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_oversampling_tuned)
        self.plot_graphs(X_test, y_test, y_pred, lgbm_model)

        X_train, X_test, y_train, y_test, X_otrain, y_otrain = self.data_prep(data, 'fs4.sav', retrain=retrain)
        filename = './models/models_new/lgbm_dataset4_tuned_l1regularized.sav'
        if retrain:
            lgbm_model = LGBMClassifier(reg_alpha=0.001, learning_rate=0.6960720013050441, max_depth= 6, \
            n_estimators= 1181, num_leaves= 69, subsample= 0.3646666061385309, random_state=123).fit(X_otrain, y_otrain)
            pickle.dump(lgbm_model, open(filename, 'wb'))
        lgbm_model = pickle.load(open(filename, 'rb'))
        y_pred = lgbm_model.predict(X_test)
        print("Evaluation metrics for LGBM with dataset 4 with l1 regularization:")
        report_l1regularized = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_l1regularized)
        self.plot_graphs(X_test, y_test, y_pred, lgbm_model)

        X_train, X_test, y_train, y_test, X_otrain, y_otrain = self.data_prep(data, 'fs4.sav', retrain=retrain)
        filename = './models/models_new/lgbm_dataset4_tuned_l2regularized.sav'
        if retrain:
            lgbm_model = LGBMClassifier(reg_lambda=1, learning_rate=0.6960720013050441, max_depth= 6, \
            n_estimators= 1181, num_leaves= 69, subsample= 0.3646666061385309, random_state=123).fit(X_otrain, y_otrain)
            pickle.dump(lgbm_model, open(filename, 'wb'))
        lgbm_model = pickle.load(open(filename, 'rb'))
        y_pred = lgbm_model.predict(X_test)
        print("Evaluation metrics for LGBM with dataset 4 with l2 regularization:")
        report_l2regularized = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        display(report_l2regularized)
        self.plot_graphs(X_test, y_test, y_pred, lgbm_model)

        # return report_subset4, report_oversampling_tuned, report_l1regularized, report_l2regularized




            

            


        
