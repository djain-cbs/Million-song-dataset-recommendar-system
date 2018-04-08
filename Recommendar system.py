# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 23:47:26 2018

@author: Deepam Jain
"""

import numpy as np
import pandas as pd

#f = open('C:/Users/Deepam Jain/Documents/GitHub/Million-song-dataset/kaggle_visible_evaluation_triplets.txt', 'r')

column_name = ['user_id','song','played']
data = pd.read_csv('C:/Users/Deepam Jain/Documents/GitHub/Million-song-dataset/kaggle_visible_evaluation_triplets.txt',sep='\t',names=column_name)

data=data[1:100000]

data['user_id'] = data['user_id'].astype('category')
data['user_id'] = data['user_id'].cat.codes

data['song'] = data['song'].astype('category')
data['song'] = data['song'].cat.codes

data.user_id.max()-data.user_id.min()

data.shape
nuser = data.user_id.nunique()
nsong = data.song.nunique()


from sklearn.cross_validation import train_test_split
train_data, test_data = train_test_split(data, test_size=0.25)


#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((nuser, nsong))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  

test_data_matrix = np.zeros((nuser, nsong))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]


from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred


item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

import scipy.sparse as sp
from scipy.sparse.linalg import svds

#get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))