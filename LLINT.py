
# from main import get_dataloader
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.multiclass import OutputCodeClassifier

#setting up arg values

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from bend_utils import get_proj_matrix, get_cos_neighbors_tensor
# from main import retrieve_with_embeddings
# from main import get_proj_matrix

def get_INLP_P(dataset_embeddings, labels, num_seps=25, max_comp=900, logreg=True):
    accuracies = []
    seps = []
    trainset = dataset_embeddings
    Trainset = trainset
    np.random.seed(0)
    print(set(labels))
    smallest_cat = min([list(labels).count(i) for i in set(labels)])
    print(smallest_cat)
    equalized_indices = []
    for i in set(labels):
        equalized_indices.extend(np.random.choice(np.where(np.array(labels) == i)[0], smallest_cat))
    trainset = trainset[equalized_indices]
    print(len(trainset))
    labels = np.array([labels[i] for i in equalized_indices])
    Trainset = trainset

    for i in tqdm(range(num_seps)):

        #print(f"fitting regressor {i}")
        if logreg:
            #lr = OutputCodeClassifier(LogisticRegression(random_state=0), code_size=0.5, random_state=0)
            lr = LogisticRegression(max_iter=10000)
        #lr = LinearSVC(max_iter=10000)
        else:
            lr = LinearSVC(max_iter=100000)
        lr.fit(trainset, labels)
        # coefs = np.concatenate([lr.estimators_[i].coef_ for i in range(len(lr.estimators_))])
        # print("RANK:", np.linalg.matrix_rank(coefs))
        seps.extend([lr.coef_[i] for i in range(len(lr.coef_))])
        #evaluate lr on train set
        y_pred = lr.predict(trainset)
        y_true = labels
        #print("train acc", np.mean(np.round(y_pred) == y_true))
        accuracies.append(np.mean(np.round(y_pred) == y_true))

        P = get_proj_matrix(torch.tensor(seps))
        trainset = np.matmul(Trainset, P.T)
        trainset = F.normalize(trainset, dim=-1)
        if type(trainset) is not torch.Tensor:
            trainset = torch.tensor(trainset)
        trainset = trainset.float()

        y_pred = lr.predict(trainset)
        if len(accuracies) > 5 and np.abs(accuracies[-1] - accuracies[-4]) < 1e-5:
            break
    print(accuracies)
    P = get_proj_matrix(torch.tensor(seps), min(len(seps), max_comp))
    return torch.tensor(P).float()
# get_INLP_P()
def get_debiased_embeddings_INLP(query_embeddings, ref_dataset_embeddings, labels, P_init=None):
    text_embeddings = query_embeddings
    text_embeddings_i = query_embeddings
    _ref_image_embeddings = (ref_dataset_embeddings / 
                        ref_dataset_embeddings.norm(dim=-1, keepdim=True))
    if P_init == None:
        P_init = get_INLP_P(_ref_image_embeddings, labels).T
    text_embeddings_i = np.matmul(text_embeddings_i, P_init)
    text_embeddings_i = F.normalize(text_embeddings_i, dim=-1)
    text_embeddings_i = torch.tensor(text_embeddings_i).float()
    print("SHAPE", ref_dataset_embeddings.shape)
    #Get the images closest to each prompt
    Ps = [P_init for i in range(len(text_embeddings))]
    for radius in (50000,3000):#(7500, 1000):
        top500 = get_cos_neighbors_tensor(text_embeddings_i,_ref_image_embeddings,radius)
        topimgs = [_ref_image_embeddings[top500[:,pnum],:]
            for pnum in range(top500.shape[1])] #top 500 images for each prompt (e/a embedding is a row), prompt # is first index

        Ps = [get_INLP_P(topimgs[pnum], 
                [labels[i] for i in top500[:,pnum]], num_seps=15,
                logreg = True).T@Ps[pnum]
                    for pnum in range(len(topimgs))] #get projection matrix for each prompt
        text_embeddings_i = torch.vstack([np.matmul(text_embeddings[i,:], Ps[i].T) for i in range(len(text_embeddings))])
        text_embeddings_i = F.normalize(text_embeddings_i, dim=-1)
        text_embeddings_i = torch.tensor(text_embeddings_i).float()
        text_embeddings = torch.vstack([np.matmul(text_embeddings[i,:], Ps[i].T) for i in range(len(text_embeddings))])
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    text_embeddings = torch.tensor(text_embeddings).float()
    return text_embeddings