import os, pickle
import pandas as pd
import numpy as np
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import missingno as msno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

import sklearn
from sklearn.feature_selection import SelectPercentile, f_classif

from src.config import Config

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
class Analysis(Config):

    def __init__(self):
        self.data = {}

    def read_file(self, fname=None):
        try:
            if fname is None:
                fname = os.path.join(Config.DATA["INPUT_PATH"])
            print("Reading file: {} ...".format(fname))
            data = pd.read_csv(fname)
            for col in data.columns: 
                if len(data[col].unique()) < 20 or col in ["12", "64", "95", "target"]: 
                    data[col] = data[col].astype("category")
            print("Data import complete for file: {} ...".format(fname))
            return data

        except FileNotFoundError:
            print(fname)
            print("File {} is not found ... Please specify the correct path in config.py".format(fname))
            

    def summary_statistics(self, data, dtype):
        if dtype == "numerical":
            df_stats_num = data.select_dtypes(["float", "int"]).describe()

            kurtosis_list = []
            skewness_list = []
            numerical_column_list = [col for col in df_stats_num]
            for col in df_stats_num:
                kurtosis_list.append(data[col].kurtosis())
                skewness_list.append(data[col].skew())

            new_dict_kurtosis = dict(zip(numerical_column_list,kurtosis_list))
            new_dict_skewness = dict(zip(numerical_column_list,skewness_list))

            new_rows_kurtosis = pd.Series(data = new_dict_kurtosis, name='kurtosis')
            new_rows_skewness = pd.Series(data = new_dict_skewness, name='skewness')

            # Append the series of kurtosis and skewness to the .describe() dataframe
            df_stats_num = df_stats_num.append(new_rows_kurtosis, ignore_index=False)
            df_stats_num = df_stats_num.append(new_rows_skewness, ignore_index=False)

            if (len(data) > 10):
                df_stats_num = pd.DataFrame(df_stats_num.transpose())
                
            # Set skewness and kurtosis type
            df_stats_num.loc[df_stats_num['kurtosis'] < 3 , 'kurtosis type'] = 'Platykurtic' # thin tails
            df_stats_num.loc[df_stats_num['kurtosis'] == 3 , 'kurtosis type'] = 'Normal - Mesokurtic'
            df_stats_num.loc[df_stats_num['kurtosis'] > 3 , 'kurtosis type'] = 'Leptokurtic' # heavy tails
            df_stats_num.loc[df_stats_num['skewness'] < 0, 'skewness type'] = 'Negatively Skewed'
            df_stats_num.loc[df_stats_num['skewness'] == 0, 'skewness type'] = 'Symmetrical'
            df_stats_num.loc[df_stats_num['skewness'] > 0, 'skewness type'] = 'Positively Skewed'
            df_stats_num.loc[(df_stats_num['skewness'] > -0.5) & (df_stats_num['skewness'] < 0.5), 'skewness lvl'] \
                = 'Fairly Symmetrical'
            df_stats_num.loc[(df_stats_num['skewness'] > -1.0) & (df_stats_num['skewness'] < -0.5) , 'skewness lvl'] \
                = 'Moderately Skewed'
            df_stats_num.loc[(df_stats_num['skewness'] > 0.5) & (df_stats_num['skewness'] < 1.0), 'skewness lvl'] \
                = 'Moderately Skewed'
            df_stats_num.loc[(df_stats_num['skewness'] > 1.0) | (df_stats_num['skewness'] < -1.0), 'skewness lvl'] \
                = 'Highly Skewed'
            final_df = df_stats_num

        elif dtype == "categorical":
            df_stats_cat = data.select_dtypes(["category"]).describe()
            if (len(data) > 10):
                df_stats_cat = pd.DataFrame(df_stats_cat.transpose())
            final_df = df_stats_cat

        return final_df

    
    def categorical_barplot(self, data, col, xlabel, title, type="standard"): 

        fig, ax = plt.subplots(figsize=(15, 5))
        if type == "standard":
            try:
                cat_index = np.unique(data[col], return_counts=True)[0]
                cat_df = pd.DataFrame(np.unique(data[col], return_counts=True)[1], index=cat_index)
                y = list(cat_df[0])
            except:
                cat_df = pd.DataFrame(data[col].value_counts())
                y = cat_df.iloc[:,0]
            x = list(cat_df.index)
        elif type == "missing": 
            x = list(data[col].index)
            y = list(data[col])
        ax.bar(x, y, color=['grey', 'red', 'green', 'blue', 'cyan'])
        for i in range(len(x)):
            ax.text(i, y[i], y[i], ha = 'center')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(col, fontsize=14)
        
        return fig

    def data_scaling(self, data):

        X = data.loc[:, ~data.columns.isin(['target'])].values
        y = data.loc[:,['target']].values
        X = pd.DataFrame(StandardScaler().fit_transform(X))
        normalized_data= pd.concat([X, pd.DataFrame(y)], axis=1)

        return X


    def boxplot(self, X, col, start_col, end_col):

        if col == 0:
            fig, ax = plt.subplots(figsize=(20,8))
            sns.boxplot(x="variable", y="value", data=pd.melt(X.iloc[:,:col+11]), ax=ax)
        else: 
            fig, ax = plt.subplots(figsize=(20,8))
            sns.boxplot(x="variable", y="value", data=pd.melt(X.iloc[:,start_col:end_col]), ax=ax)
            start_col = end_col
            end_col = end_col+11
                
        return fig, start_col, end_col


    def control_chart(self, data, col, filter=None, type='x'):

        if col != "target":
            np.random.seed(Config.ANALYSIS_CONFIG["RANDOM_SEED"])
            x = data.loc[:,col]
            MR = [np.nan]

            # Get and append moving ranges
            i = 1
            for _ in range(1, len(x)):
                MR.append(abs(x[i] - x[i-1]))
                i += 1
            MR = pd.Series(MR)

            # Concatenate mR Series with and rename columns
            data_plot = pd.concat([x,MR, data.target], axis=1)
            data_plot.columns = ["x", "mR", "target"]
            if filter is not None: 
                temp_plot = data_plot[filter:].reset_index(drop=True)
            else: 
                temp_plot = data_plot

            # Plot x and mR charts
            fig, axs = plt.subplots(1, figsize=(15,7), sharex=True)

            # x chart
            if type == "x":
                xchart = axs.scatter(temp_plot.index, temp_plot['x'], linestyle='-', marker='o', c=temp_plot['target'])
                axs.axhline(statistics.mean(data_plot['x']), color='blue')
                axs.axhline(statistics.mean(data_plot['x']) + \
                    3*statistics.mean(data_plot['mR'][1:len(data_plot['mR'])])/1.128, color = 'red', linestyle = 'dashed')
                axs.axhline(statistics.mean(data_plot['x']) - \
                    3*statistics.mean(data_plot['mR'][1:len(data_plot['mR'])])/1.128, color = 'red', linestyle = 'dashed')
                axs.set_title('X-chart for column: {}'.format(col))
                axs.legend(*xchart.legend_elements())
                axs.set(xlabel='Unit', ylabel='Value')

            # mR chart
            elif type == "mR":
                mRchart = axs.scatter(temp_plot.index, temp_plot['mR'], linestyle='-', marker='o', c=temp_plot['target'])
                axs.axhline(statistics.mean(data_plot['mR'][1:len(data_plot['mR'])]), color='blue')
                axs.axhline(statistics.mean(data_plot['mR'][1:len(data_plot['mR'])]) + \
                    3*statistics.mean(data_plot['mR'][1:len(data_plot['mR'])])*0.8525, color='red', linestyle ='dashed')
                axs.axhline(statistics.mean(data_plot['mR'][1:len(data_plot['mR'])]) - 
                    3*statistics.mean(data_plot['mR'][1:len(data_plot['mR'])])*0.8525, color='red', linestyle ='dashed')
                axs.set_ylim(bottom=0)
                axs.set_title('mR Chart for column: {}'.format(col))
                axs.legend(*mRchart.legend_elements())
                axs.set(xlabel='Unit', ylabel='Range')

            return fig


    def outlier_identification(self, data, selected_cols, mode='feature_engineering'):

        for col in selected_cols: 
            x = data.loc[:,col]
            MR = [np.nan]
            i = 1
            for _ in range(1, len(x)):
                MR.append(abs(x[i] - x[i-1]))
                i += 1
            MR = pd.Series(MR)   
            temp_data = pd.concat([x, MR, data.target], axis=1)
            temp_data.columns = ["x", "mR", "target"]
            ucl = statistics.mean(temp_data['x'])+3*statistics.mean(temp_data['mR'][1:len(temp_data['mR'])])/1.128
            lcl = statistics.mean(temp_data['x'])-3*statistics.mean(temp_data['mR'][1:len(temp_data['mR'])])/1.128

            if mode == 'feature_engineering':
                # We flag out the data points that lie outside the ucl and lcl
                # Assumption: Target is not available for prediction
                data[col+"_flag"] = ((temp_data["x"] < lcl) | (temp_data["x"] > ucl))*1
                data[col+"_flag"] = data[col+"_flag"].astype('category')

            elif mode == 'outlier_removal':
                # Remove outliers if data belongs to majority class
                mask = ((temp_data["x"] < lcl) | (temp_data["x"] > ucl)) & (temp_data["target"].astype("int") == 0)
                if mask.any():
                    temp_data.loc[mask,"x"] = np.nan
                    data[col] = temp_data["x"]

        return data


    def missingness_heatmap(self, data):

        df_missing = data.loc[:, data.isna().any()]
        df_missing = df_missing.isna()
        missing_cor = df_missing.corr(method='kendall')
        mask = np.triu(np.ones_like(missing_cor, dtype=bool))
        mask_df = missing_cor.mask(mask)
        check = [c for c in mask_df.columns if any(mask_df[c] > 0.1)]
        pair = []
        correlation = []
        if len(check) > 0:
            for col in mask_df.columns:
                for index in mask_df.index:
                    if mask_df.loc[index, col] >= 0.4:
                        pair.append(str(index+" & "+ col))
                        correlation.append(np.round(mask_df.loc[index, col], 2))

        df = pd.DataFrame({'pair': pair, 'correlation': correlation})
        df.sort_values(by="correlation", ascending=False, inplace=True)
        return df

    def missingness_analysis(self, data, type="matrix"):
        """
        Display missing data analysis matrix chart and missing data heatmap.

        Args:
        data (dataframe): Output from read_input()

        """

        missing_col = data.isnull().sum()
        percent_missing_col = round(missing_col * 100 / len(data), 2)
        fig, ax = plt.subplots(figsize=(15, 5))
        if type == "matrix":
            msno.matrix(data, ax=ax)
        elif type == "bar":
            msno.bar(data, ax=ax)
        return fig

    def missingness_class(self, data):

        class0 = data.loc[data.target==0]
        missing_data_class0 = pd.DataFrame(class0.isna().sum()[class0.isna().sum() != 0], columns=["class_0"])
        class1 = data.loc[data.target==1]
        missing_data_class1 = pd.DataFrame(class1.isna().sum()[class1.isna().sum() != 0], columns=["class_1"])
        class2 = data.loc[data.target==2]
        missing_data_class2 = pd.DataFrame(class2.isna().sum()[class2.isna().sum() != 0], columns=["class_2"])
        class3 = data.loc[data.target==3]
        missing_data_class3 = pd.DataFrame(class3.isna().sum()[class3.isna().sum() != 0], columns=["class_3"])
        class4 = data.loc[data.target==4]
        missing_data_class4 = pd.DataFrame(class4.isna().sum()[class4.isna().sum() != 0], columns=["class_4"])
        final_df = pd.concat([missing_data_class0, missing_data_class1, missing_data_class2, missing_data_class3,\
             missing_data_class4], axis=1)
            
        fig, ax = plt.subplots(figsize=(15, 5))
        colors = ['grey', 'red', 'green', 'blue', 'cyan']
        final_df.plot.bar(stacked=True, 
                                color=colors, 
                                figsize=(10,7), 
                                ax=ax, 
                                title = "Missingness Count by Target Class",
                                xlabel = "Input Variables",
                                ylabel= "Missingness Count",
                                fontsize=14)

        return fig

    def missingness_correlation(self, data):

        high_cor_missing = self.missingness_heatmap(data)
        
        if len(high_cor_missing) > 0:
            print('Column pairs with similar pattern of missingness:- \n')
            return msno.heatmap(data)

        else:
            if data.isnull().sum().sum() == 0:
                print('There are no missing data in the columns.')
            else:
                print('There is only one column that has missing data, therefore no coorelation can be done.')
        
 
    def mice_imputation(self, data):

        MICE_imputer = IterativeImputer(random_state=Config.ANALYSIS_CONFIG["RANDOM_SEED"])
        imputed_df = MICE_imputer.fit_transform(data)

        return imputed_df

    def data_transformation(self, data):

        summary_numerical = self.summary_statistics(data, "numerical")
        filter_data = data.loc[:, ~data.columns.isin(Config.ANALYSIS_CONFIG["BITRIMODAL_DISTRIBUTION"])] 
        sym_data = data.loc[:, data.columns.isin(summary_numerical[summary_numerical["skewness lvl"] ==\
                "Fairly Symmetrical"].index)]
        mskew_data = filter_data.loc[:, filter_data.columns.isin(summary_numerical[summary_numerical["skewness lvl"] \
            == "Moderately Skewed"].index)]
        hskew_data = filter_data.loc[:, filter_data.columns.isin(summary_numerical[summary_numerical["skewness lvl"] \
            == "Highly Skewed"].index)]

        mpskew_data = mskew_data.loc[:,(mskew_data>=0).all()]
        mpskew_tdata = mpskew_data.copy()
        for col in mpskew_data.columns:
            mpskew_tdata["{}_sqrt".format(col)] = np.sqrt(mpskew_data.loc[:,col])
        mnskew_data = mskew_data.loc[:,(mskew_data<0).any()]
        mnskew_tdata = mnskew_data.copy()
        for col in mnskew_data.columns:
            mnskew_tdata["{}_sqrt".format(col)] = np.sqrt(max(mnskew_data.loc[:, col]+1) - mnskew_data.loc[:, col])

        hpskew_data = hskew_data.loc[:,(hskew_data>=0).all()]
        hpskew_tdata = hpskew_data.copy()
        for col in hpskew_data.columns:
            hpskew_tdata["{}_log".format(col)] = np.log(hpskew_data.loc[:,col])
        hnskew_data = hskew_data.loc[:,(hskew_data<0).any()]
        hnskew_tdata = hnskew_data.copy()
        for col in hnskew_data.columns:
            hnskew_tdata["{}_log".format(col)] = np.log(max(hnskew_data.loc[:, col]+1) - hnskew_data.loc[:, col])

        combined_dict = dict(
            SYMMETRICAL_DATA       = sym_data,
            MODPOS_ORIGINAL        = mpskew_data,
            MODNEG_ORIGINAL        = mnskew_data,
            HIGHPOS_ORIGINAL       = hpskew_data,
            HIGHNEG_ORIGINAL       = hnskew_data,
            MODPOS_TRANSFORMED     = mpskew_tdata.loc[:, mpskew_tdata.columns.str.contains("sqrt")],
            MODNEG_TRANSFORMED     = mnskew_tdata.loc[:, mnskew_tdata.columns.str.contains("sqrt")],
            HIGHPOS_TRANSFORMED    = hpskew_tdata.loc[:, hpskew_tdata.columns.str.contains("log")],
            HIGHNEG_TRANSFORMED    = hnskew_tdata.loc[:, hnskew_tdata.columns.str.contains("log")],
            TARGET                 = data[["target"]]
        )
        combined_df = pd.concat([df for k, df in combined_dict.items()], axis=1)
        transform_numerical = self.summary_statistics(combined_df, "numerical")

        return combined_dict, transform_numerical


    def histogram_plot(self, data, type="before", grid_cols = 5):

        if type == "after":
            
            combined_dict, _ = self.data_transformation(data)
            mskew_original = pd.concat([combined_dict["MODPOS_ORIGINAL"], combined_dict["MODNEG_ORIGINAL"]], axis=1)
            mskew_transformed = pd.concat([combined_dict["MODPOS_TRANSFORMED"], combined_dict["MODNEG_TRANSFORMED"]], \
                axis=1)
            hskew_original = pd.concat([combined_dict["HIGHPOS_ORIGINAL"], combined_dict["HIGHNEG_ORIGINAL"]], axis=1)
            hskew_transformed = pd.concat([combined_dict["HIGHPOS_TRANSFORMED"], combined_dict["HIGHNEG_TRANSFORMED"]],\
                axis=1)
            original_list = [mskew_original, hskew_original]
            transformed_list = [mskew_transformed, hskew_transformed]
            skew_name = ["Moderately Skewed", "Highly Skewed"]

            for k, df in enumerate(original_list):
                print("Histogram plots before and after data transformation for {} variables:".format(skew_name[k].lower()))
                fig = plt.figure(figsize=(20,int(len(original_list[k].columns))*3))
                spec = GridSpec(ncols=2, nrows=int(len(original_list[k].columns)), figure=fig)
                
                counter = 0
                for i, tup in enumerate(original_list[k].iteritems()):
                    df = list(tup)[1]
                    ax = plt.subplot(spec[counter, 0])
                    df.hist(grid=False, bins=30, color='#00B1A9', alpha=0.3, ax=ax)
                    ax.axvline(x=df.mean(), lw=2.5, ls=':', color='red')
                    ax.axvline(x=df.median(), lw=2, ls='--', color='purple')
                    ax.set_title("Histogram for variable {} before transformation".format(original_list[k].columns[i]))
                    ax.legend(["mean", "median"])
                    counter += 1

                counter = 0
                for j, tup in enumerate(transformed_list[k].iteritems()):
                    df = list(tup)[1]
                    ax = plt.subplot(spec[counter, 1])
                    df.hist(grid=False, color='blue', bins=30, ax=ax, alpha=0.3)
                    ax.axvline(x=df.mean(), lw=2.5, ls=':', color='red')
                    ax.axvline(x=df.median(), lw=2, ls='--', color='purple')
                    ax.set_title("Histogram for variable {} after transformation".format(transformed_list[k].columns[j]))
                    ax.legend(["mean", "median"])
                    counter += 1
                
                fig.tight_layout()
                display(fig)
                    

        elif type == "before":

            summary_numerical = self.summary_statistics(data, "numerical")
            sym_data = data.loc[:, data.columns.isin(summary_numerical[summary_numerical["skewness lvl"] ==\
                "Fairly Symmetrical"].index)]
            mskew_data = data.loc[:, data.columns.isin(summary_numerical[summary_numerical["skewness lvl"] ==\
                "Moderately Skewed"].index)]
            hskew_data = data.loc[:, data.columns.isin(summary_numerical[summary_numerical["skewness lvl"] == \
                "Highly Skewed"].index)]
            skew_list = [sym_data, mskew_data, hskew_data]
            skew_name = ["Fairly Symmetrical", "Moderately Skewed", "Highly Skewed"]
            
                
            for k, df in enumerate(skew_list):
                print("Histogram plots for {} variables:".format(skew_name[k].lower()))
                fig = plt.figure(figsize=(20,int(len(skew_list[k].columns))*3))
                spec = GridSpec(ncols=grid_cols, nrows=int(len(skew_list[k].columns)), figure=fig)
                counter = 0
                j = 0
                for i, tup in enumerate(skew_list[k].iteritems()):
                    df = list(tup)[1]
                    ax = plt.subplot(spec[counter,j])
                    df.hist(grid=False, bins=30, color='#00B1A9', alpha=0.3, ax=ax)
                    ax.axvline(x=df.mean(), lw=2.5, ls=':', color='red')
                    ax.axvline(x=df.median(), lw=2, ls='--', color='purple')
                    ax.set_title("Histogram for variable {}".format(skew_list[k].columns[i]))
                    ax.legend(["mean", "median"])
                    j += 1
                    if j == grid_cols:
                        counter += 1
                        j = 0

                fig.tight_layout()
                display(fig)


    def pca_transformation(self, data, retrain=False, fname=None, save=False):

        x = data.loc[:, ~data.columns.isin(['target'])].values
        y = data.loc[:, ['target']].values
        x = StandardScaler().fit_transform(x)
        fpath = fname 
        if retrain: 
            pca = PCA(random_state=123).fit(x)
            
            # Plot
            fig = plt.figure(figsize=(10,8))
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance')
            plt.title('Explained variance ratio per principle component (Before Model Training)')
            display(fig)
            pca_variance = np.cumsum(pca.explained_variance_ratio_)
            
            # Train PCA
            index = np.where(pca_variance > 0.95)[0][0]
            pca = PCA(n_components=index, random_state=123)
            
            # Save model
            if save:
                pickle.dump(pca, open(fpath, 'wb'))
                
        # Load and run pca
        pca = pickle.load(open(fpath, "rb"))
        pcs = pca.fit_transform(x)
        
        # Plot
        fig = plt.figure(figsize=(10,8))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.title('Explained variance ratio per principle component (After Model Training)')
        display(fig)
        
        # Prep data
        columns = ['pc'+str(i+1) for i in range(pcs.shape[1])]
        pcs_df = pd.DataFrame(data=pcs, columns = columns)
        target_df = pd.DataFrame(data=y, columns=['target'])
        pca_df = pd.concat([pcs_df, target_df], axis=1)
        return pca_df


    def pca_plot(self, data):

        np.random.seed(Config.ANALYSIS_CONFIG["RANDOM_SEED"])
        for i, col in enumerate(data.loc[:,~data.columns.isin(["target"])].columns[:5]):
            if i < data.shape[1]:
                fig, ax = plt.subplots(figsize=(12,8))
                sns.scatterplot(x="pc{}".format(i+1), 
                    y="pc{}".format(i+2),
                    hue="target",
                    data=data,
                    legend="full",
                    palette= "deep",
                    style= "target",
                    size= 'target',
                    ax=ax
                )
            display(fig)

    def anova_feature(self, data):

        x = data.loc[:, ~data.columns.isin(['target'])].values
        y = data.loc[:,['target']].values
        x = StandardScaler().fit_transform(x)
        selector = SelectPercentile(score_func=f_classif, \
            percentile=Config.ANALYSIS_CONFIG["PERCENTILE_THRESHOLD"]).fit(x, y)
        anova_selected_features = np.array(data.loc[:, ~data.columns.isin(['target'])].columns)[selector.get_support()]
       
        return anova_selected_features




    