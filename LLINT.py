
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import util

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.multiclass import OutputCodeClassifier, OneVsRestClassifier


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from bend_utils import get_proj_matrix, get_cos_neighbors_tensor

def get_INLP_P(dataset_embeddings, labels, num_seps=25, max_comp=900, logreg=True):
    '''Perform iterative projection to make a dataset not linearly separable
    Args:
        dataset_embeddings: numpy array of embeddings of the dataset
        labels: list of labels for each embedding
        num_seps: max number separators to fit + project out (Default 25)
        max_comp: max number of components to keep in the projection matrix (Default 900)
        logreg: whether to use logistic regression (True) or linear SVM (False) (Default True)
    Returns:
        (P, seps)
            - P: projection matrix as a torch tensor dtype float (shape: embedding_dim x embedding_dim)
            - seps: list of separators that were fitted   
    '''
    
    accuracies = [] #store accuracies of the linear regressor at each iteration
    seps = [] #store the separators that were fitted
    trainset = dataset_embeddings
    Trainset = trainset #Trainset (capital T) is the original dataset, 
    #trainset (lowercase t) will be repeatedly having all fitted seps projected out, but we need original to refer to for proj every time
    np.random.seed(0)

    ####### 
    # Equalize the number of each spurious class (gender, race, etc.) in the input dataset
    #######
    #this is done to enable better fitting of the separators (if, e.g. 60% were female, then any other separator with >50, but <60% accuracy couldn't be fit)
    smallest_cat = min([list(labels).count(i) for i in set(labels)])
    print(smallest_cat)
    equalized_indices = []
    for i in set(labels):
        equalized_indices.extend(np.random.choice(np.where(np.array(labels) == i)[0], smallest_cat))
    trainset = trainset[equalized_indices]
    print(len(trainset))
    labels = np.array([labels[i] for i in equalized_indices])
    Trainset = trainset

    #######
    # Iteratively fit separators and project out
    #######
    for i in tqdm(range(num_seps)):
        if logreg:
            if len(set(labels)) > 2: 
                #for multiclass, use OutputCodeClassifier to reduce number of dimensions projected out to log(# categories), and maintain meaning of query
                lr = OneVsRestClassifier(LogisticRegression(max_iter=10000))#OutputCodeClassifier(LogisticRegression(random_state=0), code_size=0.8, random_state=0)
            else:
                lr = LogisticRegression(max_iter=10000)
        else: 
            #linear SVC is faster, but less accurate and seems to be worse for debiasing
            lr = LinearSVC(max_iter=100000)

        lr.fit(trainset, labels)

        if len(set(labels)) > 2: #OutputCodeClassifier has a different output format
            coefs = np.concatenate([lr.estimators_[i].coef_ for i in range(len(lr.estimators_))])
            seps.extend([coefs[i] for i in range(len(coefs))])
        else:
            seps.extend([lr.coef_[i] for i in range(len(lr.coef_))])

        y_pred = lr.predict(trainset)
        y_true = labels

        accuracies.append(np.mean(np.round(y_pred) == y_true))

        P = get_proj_matrix(torch.tensor(seps))
        trainset = np.matmul(Trainset, P.T)
        trainset = F.normalize(trainset, dim=-1)
        if type(trainset) is not torch.Tensor:
            trainset = torch.tensor(trainset)
        trainset = trainset.float()

        y_pred = lr.predict(trainset)
        if len(accuracies) > 5 and np.abs(accuracies[-1] - 1/len(set(labels))) < 1e-5:
            break
    print(accuracies)
    P = get_proj_matrix(torch.tensor(seps), min(len(seps), max_comp))
    return (torch.tensor(P).float(), seps)

def full_LLINT_debias(query_embeddings, ref_dataset_embeddings, labels, P_init=None, seps_init=[]):
    text_embeddings = query_embeddings
    text_embeddings_i = query_embeddings
    _ref_image_embeddings = (ref_dataset_embeddings / 
                        ref_dataset_embeddings.norm(dim=-1, keepdim=True))
    if P_init == None:
        P_init, seps_init = get_INLP_P(_ref_image_embeddings, labels)
        P_init = P_init.T

    text_embeddings_i = np.matmul(text_embeddings_i, P_init)
    text_embeddings_i = F.normalize(text_embeddings_i, dim=-1)
    text_embeddings_i = torch.tensor(text_embeddings_i).float()

    num_prompts = len(text_embeddings)
    #Get the images closest to each prompt
    Ps = [P_init for _ in range(num_prompts)]
    seps = [[seps_init] for _ in range(num_prompts)]
    #int(_ref_image_embeddings.shape[0]/2)
    for radius in (int(_ref_image_embeddings.shape[0]/2),1500):#(7500, 1000):
        print(_ref_image_embeddings.shape)
        top500 = get_cos_neighbors_tensor(text_embeddings_i,_ref_image_embeddings,radius)
        topimgs = [_ref_image_embeddings[top500[:,pnum],:]
            for pnum in range(top500.shape[1])] #top 500 images for each prompt (e/a embedding is a row), prompt # is first index

        locals = [get_INLP_P(topimgs[pnum], 
                [labels[i] for i in top500[:,pnum]], num_seps=15,
                logreg = True)
                    for pnum in range(len(topimgs))]
        
        seps_i = np.array(locals[0][1])
        query_sims = util.cos_sim(query_embeddings, torch.from_numpy(seps_i).float())
        print("QUERY SIMS LLINTING", query_sims)
        print(np.argsort(np.abs(query_sims)))
        seps_i = seps_i[np.argsort(np.abs(query_sims))[0], :]
        print(seps_i.shape)
        P = get_proj_matrix(torch.from_numpy(seps_i))
        #locals[pnum][0]
        Ps = [np.matmul(P,Ps[pnum])
                    for pnum in range(len(topimgs))] #get projection matrix for each prompt
        seps = [seps[i] +[seps_i] for i in range(num_prompts)]
        print("SEPS0", len(seps[0]))
        text_embeddings_i = torch.vstack([np.matmul(text_embeddings[i,:], Ps[i].T) for i in range(len(text_embeddings))])
        text_embeddings_i = F.normalize(text_embeddings_i, dim=-1)
        text_embeddings_i = torch.tensor(text_embeddings_i).float()
    text_embeddings = torch.vstack([np.matmul(text_embeddings[i,:], Ps[i].T) for i in range(len(text_embeddings))])
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    text_embeddings = torch.tensor(text_embeddings).float()
    return (text_embeddings, seps, Ps)