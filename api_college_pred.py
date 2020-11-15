from os import path
import pandas as pd
import numpy as np
from sklearn import preprocessing
import heapq
from sklearn.metrics import pairwise_distances

data = pd.read_excel("IPEDS_data.xlsx")
data_df = pd.DataFrame(data)

def preprocess():
    data = pd.read_excel("IPEDS_data.xlsx")
    data_df = pd.DataFrame(data)
    data_df = data_df[data_df["Offers Bachelor's degree"] == 'Yes']
    data_df = data_df[['Name',
                   'Percent of freshmen submitting SAT scores',
                   'Percent of freshmen submitting ACT scores',
                   'SAT Critical Reading 25th percentile score',
                   'SAT Critical Reading 75th percentile score',
                   'SAT Math 25th percentile score',
                   'SAT Math 75th percentile score',
                   'SAT Writing 25th percentile score',
                   'SAT Writing 75th percentile score',
                   'ACT Composite 25th percentile score',
                   'ACT Composite 75th percentile score']]
    data_df = data_df.dropna(thresh=2)

    data_df["SAT Critical Reading mean score"] = (data_df["SAT Critical Reading 25th percentile score"]
                                                         +data_df["SAT Critical Reading 75th percentile score"])/2
    data_df["SAT Math mean score"] = (data_df["SAT Math 25th percentile score"]
                                             +data_df["SAT Math 75th percentile score"])/2
    data_df["SAT Writing mean score"] = (data_df["SAT Writing 25th percentile score"]
                                                +data_df["SAT Writing 75th percentile score"])/2
    data_df["ACT Composite mean score"] = (data_df["ACT Composite 25th percentile score"]
                                                  +data_df["ACT Composite 75th percentile score"])/2
    data_df_25 = data_df[['Name',
                   'SAT Critical Reading 25th percentile score',
                   'SAT Math 25th percentile score',
                   'SAT Writing 25th percentile score',
                   'ACT Composite 25th percentile score']]
    data_df_75 = data_df[['Name',
                   'SAT Critical Reading 75th percentile score',
                   'SAT Math 75th percentile score',
                   'SAT Writing 75th percentile score',
                   'ACT Composite 75th percentile score']]
    data_df_mean = data_df[['Name',
                   'SAT Critical Reading mean score',
                   'SAT Math mean score',
                   'SAT Writing mean score',
                   'ACT Composite mean score']]
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
    data_norm_25 = pd.DataFrame(X_norm)
    return data_norm_25

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
    # data is the (k,4) matrix, input_valueis a (1,4) array
    sim_score = []
    sim_name = []
    name = ""
    b = np.array([train_X])
    for i in range(np.shape(data_X)[0]):  # for each college
        a = np.array([data_X.values[i]])

        cos_lib = similarity(a, b, threshold)

        name = data_y[i]
        sim_score.append(cos_lib)
        sim_name.append(name)
    sim_list = heapq.nlargest(3, zip(sim_score, sim_name))
    return sim_list  # return similiarity score and name of top 3 schools

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
    data_X = data.drop(columns = ["Name"])
    data_y = data["Name"].tolist()
    sim_result_score = []
    sim_result_name = []
    for i in range(3):
        sim_result = sim_input(data_X,data_y,input_data,threshold)[i]
        sim_result_name.append(sim_result[1])
        recommend = np.array(data.loc[data["Name"] == sim_result_name[i]])[0]
        sim_result_score.append(recommend[1:].tolist())
    return sim_result_name,sim_result_score

def prediction_school(input_data):
    data_25 = pd.DataFrame(pd.read_csv("data_df_25.csv"))
    data_mean = pd.DataFrame(pd.read_csv("data_df_mean.csv"))
    data_75 = pd.DataFrame(pd.read_csv("data_df_75.csv"))
    threshold = open('threshold.txt','r').readlines()
    threshold = [float(i) for i in threshold]
    pred1_name,pred1_score = prediction(data_25,input_data,threshold[0])
    pred2_name,pred2_score = prediction(data_mean,input_data,threshold[1])
    pred3_name,pred3_score = prediction(data_75,input_data,threshold[2])
    pred_name = pred1_name + pred2_name + pred3_name
    pred_score = pred1_score + pred2_score + pred3_score
    return pred_name,pred_score