# Core Packages for data processing
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import datetime as dt

# Quandl for retrieving data
import quandl

# ML Packages
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric

# Performace Metric Packages
from sklearn.metrics import (confusion_matrix, accuracy_score, 
classification_report, mean_squared_error)
import scipy

# Preprocessing Packages
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Used to get VIX and S&P500 indices Data
import pandas_datareader.data as web

# Used for File importing
import os



# Quantdl API Key
api_key_quandl = "Rupks5L-rNLVhyfJ2jwy"
quandl.ApiConfig.api_key = api_key_quandl

# Import Small Universe Tickers
tickers = pd.read_csv('./data/tickers.csv')

# Quandl Call to Obtain Small Universe data
stock_data = quandl.get_table('WIKI/PRICES', ticker = tickers['Ticker'], date = { 'gte': '2000-01-01', 'lte': '2019-01-10' }, paginate=True)

# Quandl Call to get 3 month treasury rate, followed by normalization
DTB3 = quandl.get('FRED/DTB3', start_date='2000-01-01', end_date='2019-01-10')
DTB3.rename(columns={'Value':'DTB3'}, inplace=True)
DTB3['DTB3'] = DTB3['DTB3']/100
DTB3['DTB3'] = DTB3['DTB3'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

# Obtaining indicies Data followed by normalization and joining to DTB data
SPY_data = web.DataReader('SPY', 'yahoo', start="2000-01-01", end="2019-01-10" )['Adj Close']
VIX_data = web.DataReader('^VIX', 'yahoo', start="2000-01-01", end="2019-01-10" )['Adj Close']
indices_data = pd.concat([SPY_data, VIX_data], axis=1)
indices_data.columns = ['SPY_adj_close', 'VIX_adj_close']
indices_data = indices_data.transform(lambda x: (x - x.min()) / (x.max() - x.min()))
indices_data = indices_data.join(DTB3)
indices_data.index.rename('date', inplace=True)


# Normalizing Appropriate Stock Data
stock_data.set_index(['ticker', 'date'], inplace=True)
stock_data.sort_index(inplace=True)
norm_stock_data = stock_data.groupby('ticker')[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']].transform(lambda x: (x - x.min()) / (x.max() - x.min()))


# Calucluting adj_close Percent Change from day to day
norm_stock_data['adj_close_pct_chg'] = norm_stock_data['adj_close'].groupby(['ticker']).pct_change()

# A positive percent change represents an increase in stock price from previous day
norm_stock_data['price_rise'] = (norm_stock_data['adj_close_pct_chg'] > 0).astype(int)


# Simple Moving Average over 5, 10, 100 days
norm_stock_data['SMA_5'] = norm_stock_data.groupby(['ticker'])['adj_close'].rolling(5).mean().reset_index(0,drop=True)
norm_stock_data['SMA_20'] = norm_stock_data.groupby(['ticker'])['adj_close'].rolling(20).mean().reset_index(0,drop=True)
norm_stock_data['SMA_100'] = norm_stock_data.groupby(['ticker'])['adj_close'].rolling(100).mean().reset_index(0,drop=True)


# Exponential Moving Average over 5, 12, 26 days
norm_stock_data['EWM_5'] = norm_stock_data.groupby('ticker').apply(lambda x: x['adj_close'].ewm(span=5).mean()).reset_index(0,drop=True)
norm_stock_data['EWM_12'] = norm_stock_data.groupby('ticker').apply(lambda x: x['adj_close'].ewm(span=12).mean()).reset_index(0,drop=True)
norm_stock_data['EWM_26'] = norm_stock_data.groupby('ticker').apply(lambda x: x['adj_close'].ewm(span=26).mean()).reset_index(0,drop=True)


# Calculating MACD as differnce of 12 day EWM and 26 day EWM
norm_stock_data['MACD'] = norm_stock_data['EWM_12'] - norm_stock_data['EWM_26']


# Calculating bollinger bands over 21 day window
norm_stock_data['BB_std'] = norm_stock_data.groupby('ticker')['adj_close'].rolling(21).std().reset_index(0,drop=True)
norm_stock_data['BB_middle_band'] = norm_stock_data.groupby('ticker')['adj_close'].rolling(21).mean().reset_index(0,drop=True)

norm_stock_data['BB_lower_band'] = norm_stock_data['BB_middle_band'] - 1.96*norm_stock_data['BB_std']
norm_stock_data['BB_upper_band'] = norm_stock_data['BB_middle_band'] + 1.96*norm_stock_data['BB_std']


# Calculating Stochastic Oscillator Signal, Indicator, and Crossover
norm_stock_data['K_low'] = norm_stock_data.groupby('ticker')['adj_close'].rolling(14).min().reset_index(0,drop=True)
norm_stock_data['K_high'] = norm_stock_data.groupby('ticker')['adj_close'].rolling(14).max().reset_index(0,drop=True)
norm_stock_data['K_indicator'] = (norm_stock_data['adj_close'] - norm_stock_data['K_low']) / (norm_stock_data['K_high'] - norm_stock_data['K_low'])
norm_stock_data['K_signal'] = norm_stock_data.groupby('ticker')['K_indicator'].rolling(3).mean().reset_index(0,drop=True)
norm_stock_data['K_crossover'] = norm_stock_data['K_indicator'] - norm_stock_data['K_signal']


# Calculating RSI components to determine RSI indicator, also normalizing RSI to be between 0-1 like rest of data
norm_stock_data['RSIchange'] = norm_stock_data.groupby('ticker')['adj_close'].diff()
norm_stock_data['RSIgain'] = norm_stock_data.RSIchange.mask(norm_stock_data.RSIchange < 0, 0.0)
norm_stock_data['RSIloss'] = -norm_stock_data.RSIchange.mask(norm_stock_data.RSIchange > 0, -0.0)
norm_stock_data['RSIavggain'] = norm_stock_data.groupby('ticker')['RSIgain'].rolling(14).mean().reset_index(0,drop=True)
norm_stock_data['RSIavgloss'] = norm_stock_data.groupby('ticker')['RSIloss'].rolling(14).mean().reset_index(0,drop=True)
norm_stock_data['RSI'] = (100- (100/(1+(norm_stock_data['RSIavggain']/norm_stock_data['RSIavgloss']))))/100


# Dropping Columns not used for ML
norm_stock_data.drop(['BB_std', 'BB_middle_band', 'K_low', 'K_high', 'RSIchange', 'RSIgain', 'RSIloss', 'RSIavggain', 'RSIavgloss'], axis = 1, inplace=True)


# Setting appropriate index and joining indicie data
norm_stock_data.reset_index(inplace=True)
norm_stock_data.set_index('date', inplace=True)
norm_stock_data = norm_stock_data.join(indices_data)


# Replacing any possible inf from calculations with NaN
norm_stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)


# Forward Filling data to prevent look ahead bias
norm_stock_data.reset_index(inplace=True)
norm_stock_data.set_index(['ticker', 'date'], inplace=True)
norm_stock_data = norm_stock_data.groupby(['ticker']).fillna(method='ffill')


# Dropping NaNs
norm_stock_data_dropna = norm_stock_data.dropna()


# Classifiers Used
dict_classifiers = {
    "Logistic_reg": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
}

performance_metric = []
for classifier_model, model_instantiation in dict_classifiers.items():
    for ticker in norm_stock_data_dropna.index.get_level_values('ticker').unique():
        # Retrieving Appropriate Ticker
        tmp_data = norm_stock_data_dropna.loc[ticker]
        
        # Splitting to Training and Testing
        X_train, X_test, y_train, y_test = train_test_split(tmp_data.loc[:, ~tmp_data.columns.isin(['price_rise'])], tmp_data['price_rise'], test_size=0.4, random_state=123)
        
        # Performing Logisitc Regression and calculating performance metric
        if (classifier_model == 'Logistic_reg'):
            model = LogisticRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
            auc = metrics.roc_auc_score(y_test, y_pred)
            f1_score = metrics.f1_score(y_test, y_pred)
            KS = scipy.stats.ks_2samp(y_test, y_pred)
            Acc = metrics.accuracy_score(y_test, y_pred)*100
            Prec = metrics.precision_score(y_test, y_pred)*100
            Rec = metrics.recall_score(y_test, y_pred)*100
            performance_metric.append([classifier_model, np.nan, ticker, conf_matrix, fpr, tpr, auc, f1_score, Acc, Prec, Rec, KS[0], KS[1]])
        
        # Performing KNN regression with in loop to maximize F1 score and AUC then calculating performance metrics
        prevAUC = 0
        prev_f1 = 0
        best_n = 0
        if (classifier_model == 'KNN'):
            for i in range(1,30):
                model = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
                y_pred = model.predict(X_test)
                auc = metrics.roc_auc_score(y_test, y_pred)
                f1_score = metrics.f1_score(y_test, y_pred)
                if (f1_score > prev_f1) and (auc > prevAUC):
                    prevAUC = auc
                    prev_f1 = f1_score
                    best_n = i
                
            model = KNeighborsClassifier(n_neighbors=best_n).fit(X_train, y_train)
            y_pred = model.predict(X_test)
            auc = metrics.roc_auc_score(y_test, y_pred)
            f1_score = metrics.f1_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
            KS = scipy.stats.ks_2samp(y_test, y_pred)
            Acc = metrics.accuracy_score(y_test, y_pred)*100
            Prec = metrics.precision_score(y_test, y_pred)*100
            Rec = metrics.recall_score(y_test, y_pred)*100
            performance_metric.append([classifier_model, best_n, ticker, conf_matrix, fpr, tpr, auc, f1_score, Acc, Prec, Rec, KS[0], KS[1]])

# Combining Performance Metrics into Dataframe
performace_metric_df = pd.DataFrame(np.vstack(np.array(performance_metric, dtype=object)))
performace_metric_df.columns = ['model', 'best_hyperparameter', 'ticker', 'Confusion_Matrix', 'fpr', 'tpr', 'AUC', 'F1', 'Accuracy', 'Precision', 'Recall', 'KS_Stat', 'KS_pvalue']
performace_metric_df = performace_metric_df.astype({'AUC': 'float64', 'F1':'float64', 'Accuracy':'float64', 'Precision':'float64', 'Recall':'float64', 'KS_Stat':'float64', 'KS_pvalue':'float64'})


# Saving Small Universe Performance Metrics to CSV
performace_metric_df.to_csv('output/small_universe_performance.csv', index=False)

# Function form of previously calculated SMA
def SMA(df):
    df['SMA_5'] = df['close'].rolling(5).mean()
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['SMA_100'] = df['close'].rolling(100).mean()
    return df


# Function form of previously calculated EWM and MACD
def MACD(df):
    df['EWM_5'] = df['close'].ewm(span=5).mean()
    df['EWM_12'] = df['close'].ewm(span=12).mean()
    df['EWM_26'] = df['close'].ewm(span=26).mean()
    df['MACD'] = df['EWM_12'] - df['EWM_26']
    return df


# Function form of previously calculated Bollinger Bands
def bollingerbands(df):
    df['BB_std'] = df['close'].rolling(21).std()
    df['BB_middle_band'] = df['close'].rolling(21).mean()
    df['BB_lower_band'] = df['BB_middle_band'] - 1.96*tmp_stock_data['BB_std']
    df['BB_upper_band'] = df['BB_middle_band'] + 1.96*tmp_stock_data['BB_std']
    df.drop(['BB_std', 'BB_middle_band'], axis = 1, inplace=True)
    return df


# Function form of previously calculated Stochastic Oscillator
def stochoscillator(df):
    df['K_low'] = df['close'].rolling(14).min()
    df['K_high'] = df['close'].rolling(14).max()
    df['K_indicator'] = (df['close'] - df['K_low']) / (df['K_high'] - df['K_low'])
    df['K_signal'] = df['K_indicator'].rolling(3).mean()
    df['K_crossover'] = df['K_indicator'] - df['K_signal']
    df.drop(['K_low', 'K_high'], axis = 1, inplace=True)
    return df


# Function form of previously calculated RSI
def RSI(df):
    df['RSIchange'] = df['close'].diff()
    df['RSIgain'] = df.RSIchange.mask(df.RSIchange < 0, 0.0)
    df['RSIloss'] = -df.RSIchange.mask(df.RSIchange > 0, -0.0)
    df['RSIavggain'] = df['RSIgain'].rolling(14).mean()
    df['RSIavgloss'] = df['RSIloss'].rolling(14).mean()
    df['RSI'] = (100- (100/(1+(df['RSIavggain']/df['RSIavgloss']))))/100
    df.drop(['RSIchange', 'RSIgain', 'RSIloss', 'RSIavggain', 'RSIavgloss'], axis=1, inplace=True)
    return df


# All tickers in large universe
nasd_tickers = pd.read_csv('data/tickers_nasd.csv', usecols=['Symbol'])
nyse_tickers = pd.read_csv('data/tickers_nyse.csv', usecols=['Symbol'])
large_universe_tickers = nyse_tickers.append(nasd_tickers)


# All tickers present in stock_dfs folder
stock_dfs_files = os.listdir(os.getcwd()+'/data/stock_dfs')
csv_files = []
for csvfile in stock_dfs_files:
    csv_files.append(csvfile[0:len(csvfile) - 4])


# Matching tickers from large universe to data present in stock_dfs folder
s = set(list(large_universe_tickers['Symbol']))
matched_tickers = [x for x in csv_files if x in s]


# Using same calculation methods as previously described in function form, then performing same ML technique over each individual stock
dict_classifiers = {
    "Logistic_reg": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
}
performance_metric = []
for mticker in matched_tickers:
#     print(mticker)
    tmp_stock_data = pd.read_csv('data/stock_dfs/{}.csv'.format(mticker), parse_dates=True, index_col=0)
    tmp_stock_data = tmp_stock_data.transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    tmp_stock_data['close_pct_chg'] = tmp_stock_data['close'].pct_change()
    tmp_stock_data['price_rise'] = (tmp_stock_data['close_pct_chg'] > 0).astype(int)
    tmp_stock_data.sort_index(inplace=True)
    tmp_stock_data = SMA(tmp_stock_data)
    tmp_stock_data = MACD(tmp_stock_data)
    tmp_stock_data = bollingerbands(tmp_stock_data)
    tmp_stock_data = stochoscillator(tmp_stock_data)
    tmp_stock_data = RSI(tmp_stock_data)
    tmp_stock_data = tmp_stock_data.join(indices_data)
    tmp_stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    tmp_stock_data = tmp_stock_data.fillna(method='ffill')
    tmp_stock_data = tmp_stock_data.dropna()
    
    X_train, X_test, y_train, y_test = train_test_split(tmp_stock_data.loc[:, ~tmp_stock_data.columns.isin(['price_rise'])], tmp_stock_data['price_rise'], test_size=0.4, random_state=123)
    
    for classifier_model, model_instantiation in dict_classifiers.items():

        if (classifier_model == 'Logistic_reg'):
            model = LogisticRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
            auc = metrics.roc_auc_score(y_test, y_pred)
            f1_score = metrics.f1_score(y_test, y_pred)
            KS = scipy.stats.ks_2samp(y_test, y_pred)
            Acc = metrics.accuracy_score(y_test, y_pred)*100
            Prec = metrics.precision_score(y_test, y_pred)*100
            Rec = metrics.recall_score(y_test, y_pred)*100
            performance_metric.append([classifier_model, np.nan, mticker, conf_matrix, fpr, tpr, auc, f1_score, Acc, Prec, Rec, KS[0], KS[1]])

        prevAUC = 0
        prev_f1 = 0
        best_n = 0
        if (classifier_model == 'KNN'):
            for i in range(1,30):
                model = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
                y_pred = model.predict(X_test)
                auc = metrics.roc_auc_score(y_test, y_pred)
                f1_score = metrics.f1_score(y_test, y_pred)
                if (f1_score > prev_f1) and (auc > prevAUC):
                    prevAUC = auc
                    prev_f1 = f1_score
                    best_n = i

            model = KNeighborsClassifier(n_neighbors=best_n).fit(X_train, y_train)
            y_pred = model.predict(X_test)
            auc = metrics.roc_auc_score(y_test, y_pred)
            f1_score = metrics.f1_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
            KS = scipy.stats.ks_2samp(y_test, y_pred)
            Acc = metrics.accuracy_score(y_test, y_pred)*100
            Prec = metrics.precision_score(y_test, y_pred)*100
            Rec = metrics.recall_score(y_test, y_pred)*100
            performance_metric.append([classifier_model, best_n, mticker, conf_matrix, fpr, tpr, auc, f1_score, Acc, Prec, Rec, KS[0], KS[1]])

performace_metric_df = pd.DataFrame(np.vstack(np.array(performance_metric, dtype=object)))
performace_metric_df.columns = ['model', 'best_hyperparameter', 'ticker', 'Confusion_Matrix', 'fpr', 'tpr', 'AUC', 'F1', 'Accuracy', 'Precision', 'Recall', 'KS_Stat', 'KS_pvalue']
performace_metric_df = performace_metric_df.astype({'AUC': 'float64', 'F1':'float64', 'Accuracy':'float64', 'Precision':'float64', 'Recall':'float64', 'KS_Stat':'float64', 'KS_pvalue':'float64'})


performace_metric_df.sort_values(['F1', 'AUC'], ascending=False, inplace=True)
performace_metric_df.reset_index(drop=True, inplace=True)


performace_metric_df_log = performace_metric_df[performace_metric_df['model'] == 'Logistic_reg']
performace_metric_df_knn = performace_metric_df[performace_metric_df['model'] == 'KNN']


performace_metric_df_log['model_rank'] = performace_metric_df_log.reset_index().index + 1
performace_metric_df_knn['model_rank'] = performace_metric_df_knn.reset_index().index + 1


performace_metric_df = performace_metric_df_log.append(performace_metric_df_knn)


performace_metric_df['model_rank_overall'] = performace_metric_df.index+1


performace_metric_df.iloc[0:20, :].to_csv('output/large_universe_performance_top20.csv', index=False)


# Saving Large Universe Performance Metrics to CSV
performace_metric_df.to_csv('output/large_universe_performance.csv', index=False)

