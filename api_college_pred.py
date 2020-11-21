from os import path
import pandas as pd
import numpy as np
from sklearn import preprocessing
import heapq
from sklearn.metrics import pairwise_distances

data = pd.read_csv("college_data_cleaned.csv")
data_df = pd.DataFrame(data)

def preprocess():
    data = pd.read_csv("college_data_cleaned.csv")
    data_df = pd.DataFrame(data)
    data_df = data_df[data_df["offers bachelor's degree"] == 'yes']
    data_df = data_df[['name',
                       'sat critical reading 25th percentile score',
                       'sat critical reading 75th percentile score',
                       'sat math 25th percentile score',
                       'sat math 75th percentile score',
                       'sat writing 25th percentile score',
                       'sat writing 75th percentile score',
                       'act composite 25th percentile score',
                       'act composite 75th percentile score']]
    data_df = data_df.dropna(thresh=2)
    data_df["sat critical reading mean score"] = (data_df["sat critical reading 25th percentile score"]
                                                  + data_df["sat critical reading 75th percentile score"]) / 2
    data_df["sat math mean score"] = (data_df["sat math 25th percentile score"]
                                      + data_df["sat math 75th percentile score"]) / 2
    data_df["sat writing mean score"] = (data_df["sat writing 25th percentile score"]
                                         + data_df["sat writing 75th percentile score"]) / 2
    data_df["act composite mean score"] = (data_df["act composite 25th percentile score"]
                                           + data_df["act composite 75th percentile score"]) / 2
    data_df_25 = data_df[['id',
                          'name',
                          'sat critical reading 25th percentile score',
                          'sat math 25th percentile score',
                          'sat writing 25th percentile score',
                          'act composite 25th percentile score']]
    data_df_75 = data_df[['id',
                          'name',
                          'sat critical reading 75th percentile score',
                          'sat math 75th percentile score',
                          'sat writing 75th percentile score',
                          'act composite 75th percentile score']]
    data_df_mean = data_df[['id',
                            'name',
                            'sat critical reading mean score',
                            'sat math mean score',
                            'sat writing mean score',
                            'act composite mean score']]
    data_df_25 = data_df_25.dropna()
    data_df_75 = data_df_75.dropna()
    data_df_mean = data_df_mean.dropna()
    data_df_25.to_csv('data_df_25.csv',index=False)
    data_df_75.to_csv('data_df_75.csv',index=False)
    data_df_mean.to_csv('data_df_mean.csv',index=False)
    return (path.exists('data_df_25.csv') and path.exists('data_df_75.csv') and path.exists('data_df_mean.csv'))

def normDF(data_X):
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
    scaler.fit(data_X)
    X_norm = scaler.transform(data_X)
    data_norm = pd.DataFrame(X_norm)
    return data_norm

#to transform features by scaling each feature to a given range, e.g. between zero and one.
def minmaxscaler_input(df,train_X):
    maxlist = df.max().to_list()
    minlist = df.min().to_list()
    input_scale = []
    for i in range(len(train_X)):
        input_scale.append((train_X[i] - minlist[i])/(maxlist[i] -minlist[i])*10)
    return input_scale

# to calculate the similarity between input score and all school's
def similarity(compare_value, train_X, threshold=1):
    # a= np.array(compare_value)
    # b=np.array(input_value)
    cos_lib = 1 - pairwise_distances(compare_value, train_X) / threshold
    cos_lib = relu1(cos_lib)
    return cos_lib


def relu1(result):
    return min(max(result[0][0], 0), 1)


#
def similarity_sort(data_X, data_y, train_X, threshold):
    # data is the (k,4) matrix, input_value is a (1,4) array
    sim_score = []
    sim_id = []
    b = np.array([train_X])
    for i in range(np.shape(data_X)[0]):  # for each college
        a = np.array([data_X.values[i]])

        cos_lib = similarity(a, b, threshold)

        sim_score.append(cos_lib)
        sim_id.append(data_y[i])
    sim_list = heapq.nlargest(3, zip(sim_score, sim_id))
    return sim_list  # return similiarity score and id of top 3 schools

def sim_input(df,data_y,input_data,threshold):
    #max_min scale the input value to fit the dataset
    input_value = np.array(minmaxscaler_input(df,input_data))
    #max_min scale the chosen dataset
    df_norm = normDF(df)
    #calculate the similaity between input and all school's data,
    #sort and return top 3 match school
    sim = similarity_sort(df_norm,data_y,input_value,threshold)
    return sim

def prediction(data,input_data,threshold):
    data_X = data.drop(columns=["id", "name"])
    data_y = data["id"].tolist()
    sim_result_score = []
    sim_result_id = []
    for i in range(3):
        sim_result = sim_input(data_X, data_y, input_data, threshold)[i]
        sim_result_id.append(sim_result[1])
        recommend = np.array(data.loc[data["id"] == sim_result_id[i]])[0]
        sim_result_score.append(recommend[1:].tolist())
    return sim_result_id, sim_result_score

def prediction_school(input_data):
    data_25 = pd.DataFrame(pd.read_csv("data_df_25.csv"))
    data_mean = pd.DataFrame(pd.read_csv("data_df_mean.csv"))
    data_75 = pd.DataFrame(pd.read_csv("data_df_75.csv"))
    threshold = open('threshold.txt', 'r').readlines()
    threshold = [float(i) for i in threshold]
    pred1_id, pred1_score = prediction(data_25, input_data, threshold[0])
    pred2_id, pred2_score = prediction(data_mean, input_data, threshold[1])
    pred3_id, pred3_score = prediction(data_75, input_data, threshold[2])
    pred_name = pred1_id + pred2_id + pred3_id
    pred_score = pred1_score + pred2_score + pred3_score
    return pred_name, pred_score