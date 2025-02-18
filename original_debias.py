import torch
import numpy as np
from transformers import CLIPModel, AutoProcessor
import pandas as pd
import torch.nn.functional as F
import yaml
import json
from datasets import load_dataset
import queries
import scipy.stats
from bend_utils import *
from LLINT import get_debiased_embeddings_INLP, get_INLP_P

config = yaml.safe_load(open("experimental_configs/celeba_hair_gender_clip-vit-base-patch16.yml"))

print(config)
############################################
_MODEL_NAME = config['model_ID'].split('/')[-1]
############################################
att_to_debias = config['att_to_debias']
dataset_name = config['dataset_name']
model_ID = config['model_ID']
query_type = config['query_type']
random_seed = config['random_seed']

if query_type == 'hair':
    query_classes = queries.hair_classes

elif query_type == 'stereotype':
    query_classes = queries.stereotype_classes

else:
    print(f'{query_type} not implemented')

print(f"query_classes: {query_classes}")

#
normalize = True
lam = 1000
#

if att_to_debias == 'race':
    if dataset_name == 'UTKFace':
        att_elements = ['White', 'Black', 'Asian', 'Indian', 'Latino Hispanic']
    else:
        att_elements = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']
elif att_to_debias == 'gender':
    att_elements = ['Male', 'Female']
else:
    print('{att_to_debias} not implemented')

############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################

model_id = config['model_ID']

vl_model = CLIPModel.from_pretrained(model_ID).to(device)
processor = AutoProcessor.from_pretrained(model_ID)

def get_embeddings(input_text : list, clip_model, clip_processor, normalize=True):

    with torch.no_grad():
        inputs = clip_processor(text=input_text, return_tensors="pt").to(device)

        query_text_embedding = clip_model.get_text_features(**inputs)#.to('cpu').numpy()

    if normalize:
        query_text_embedding /= query_text_embedding.norm(dim=-1, keepdim=True)
    return query_text_embedding

####################################################
def utk_ethnicity_map(x):
    if x == 0:
        return 'White'
    if x == 1:
        return 'Black'
    if x == 2:
        return 'Asian'
    if x == 3:
        return 'Indian'
    if x == 4:
        return 'Latino Hispanic'



def load_embedding_dataset(ds_path):
    embeddings_dataset = load_dataset("json", data_files=ds_path, split='train')
    embeddings_dataset = embeddings_dataset.with_format("np", columns=["embedding"], output_all_columns=True)

    if 'Male' in embeddings_dataset.features.keys():
        embeddings_dataset = embeddings_dataset.map(
            lambda x: {"gender": 'Male' if x['Male'] == 1 else 'Female'})
    if 'utk_race' in embeddings_dataset.features.keys():
        embeddings_dataset = embeddings_dataset.map(
            lambda x: {"race": utk_ethnicity_map(x['utk_race'])})

    if normalize:
        embeddings_dataset = embeddings_dataset.map(
            lambda x: {"embedding": x['embedding'].reshape(-1)/np.linalg.norm(x['embedding'].reshape(-1))})
    else:
        embeddings_dataset = embeddings_dataset.map(
            lambda x: {"embedding": x['embedding'].reshape(-1)})
    return embeddings_dataset

####################################################
# _embeddings_dataset = load_embedding_dataset(f'data/{dataset_name}_featurized_{_MODEL_NAME}.jsonl')
train_embeddings_dataset = load_embedding_dataset(f'data/{dataset_name}_train_featurized_{_MODEL_NAME}.jsonl')
val_embeddings_dataset = load_embedding_dataset(f'data/{dataset_name}_validation_featurized_{_MODEL_NAME}.jsonl')

####################################################

with open(f"query_templates/{query_type}_{att_to_debias}_query_templates.json", 'r') as file:
    instantiated_search_classes = json.load(file)
####################################################

with open(f"data/fold_indices/{dataset_name}_validation_featurized_{_MODEL_NAME}_folds.jsonl", 'r') as file:
    fold_dict = json.load(file)
####################################################

reference_embeddings_dataset = train_embeddings_dataset#_embeddings_dataset.select(fold_dict[str(0)]['train_indices'])
target_embeddings_dataset = val_embeddings_dataset.select(fold_dict[str(0)]['test_indices'])

####################################################
if query_type == 'hair':
    query_is_labeled = True
else:
    query_is_labeled = False

ref_spurious_class_list = att_elements
target_spurious_class_list = att_elements

gendercode = {"Male": 1, "Female": 0}
ref_gender_list = np.array(list(map(lambda g: gendercode[g], reference_embeddings_dataset["gender"])))

spurious_prompt = ["A photo of a man", "A photo of a woman"]
spurious_prompt_embedding = get_embeddings(spurious_prompt, vl_model, processor, normalize).to('cpu').numpy()
P0 = get_proj_matrix(spurious_prompt_embedding).astype(np.float32)
print(P0.shape, reference_embeddings_dataset["embedding"].shape)
P_init = get_INLP_P(F.normalize(torch.tensor(reference_embeddings_dataset["embedding"])), ref_gender_list)


result_dict = {}
for query_class in query_classes:
    result_dict[query_class] = {}

    result_dict[query_class]['Vanilla'] = {}

    result_dict[query_class]['P0'] = {}

    result_dict[query_class]['Inclusive_P0'] = {}

    result_dict[query_class]['bend_vlm'] = {}

    result_dict[query_class]['llint'] = {}

    query_class_prompt = instantiated_search_classes[query_class]['classes']
    class_prompt_embedding = get_embeddings(query_class_prompt, vl_model, processor, normalize).to('cpu').numpy()

    for k_fold in range(1):

        reference_embeddings_dataset = train_embeddings_dataset#.select(fold_dict[str(k_fold)]['train_indices'])
        target_embeddings_dataset = val_embeddings_dataset.select(fold_dict[str(k_fold)]['test_indices'])

        query_text = [instantiated_search_classes[query_class]['query']]
        print(query_class)
        print(query_text)


        spurious_prompt = instantiated_search_classes[query_class]['spurious_prompts']
        inclusive_candidate_prompt = instantiated_search_classes[query_class]['augmentations']
        S = [[0,1]]

        print(spurious_prompt)
        print(inclusive_candidate_prompt)

        spurious_att_array = np.asarray(target_embeddings_dataset[att_to_debias])
        ref_spurious_att_array = np.asarray(reference_embeddings_dataset[att_to_debias])

        spurious_att_prior = {}
        for spurious_att in target_spurious_class_list:
            spurious_att_prior[spurious_att] = spurious_att_array[spurious_att_array==spurious_att].shape[0]/spurious_att_array.shape[0]
        print(f'spurious att prior: {spurious_att_prior}')


        ref_spurious_att_prior = {}
        for r_spurious_att in ref_spurious_class_list:
            ref_spurious_att_prior[r_spurious_att] = ref_spurious_att_array[ref_spurious_att_array==r_spurious_att].shape[0]/ref_spurious_att_array.shape[0]
        print(f'ref spurious att prior: {ref_spurious_att_prior}')

        if query_is_labeled:
            eg_array = np.asarray(target_embeddings_dataset[query_class])
            conditional_spurious_att_prior = {}
            conditional_spurious_att_array = spurious_att_array[eg_array==1]
            for spurious_att in target_spurious_class_list:
                conditional_spurious_att_prior[spurious_att] = conditional_spurious_att_array[conditional_spurious_att_array==spurious_att].shape[0]/conditional_spurious_att_array.shape[0]
            print(f'conditional spurious att prior: {conditional_spurious_att_prior}')

        if query_is_labeled:
            prior_for_metric =spurious_att_prior
        else:
            prior_for_metric = spurious_att_prior




        query_text_embedding = get_embeddings(query_text, vl_model, processor, normalize).to('cpu').numpy()
        spurious_prompt_embedding = get_embeddings(spurious_prompt, vl_model, processor, normalize).to('cpu').numpy()
        inclusive_candidate_prompt_embedding = get_embeddings(inclusive_candidate_prompt, vl_model, processor, normalize).to('cpu').numpy()

        P0 = get_proj_matrix(spurious_prompt_embedding)

        M = get_M(inclusive_candidate_prompt_embedding, S)
        G = lam * M + np.eye(M.shape[0])
        inclusive_P_star = np.matmul(P0, np.linalg.inv(G))


        P0_embeddings = np.matmul(query_text_embedding, P0.T)
        P0_embeddings = F.normalize(torch.tensor(P0_embeddings), dim=-1).numpy()

        P0_class_embeddings = np.matmul(class_prompt_embedding.astype(np.float32), P0.T)
        P0_class_embeddings = F.normalize(torch.tensor(P0_class_embeddings), dim=-1).numpy().astype(np.float32)

        inclusive_P_star_embeddings = np.matmul(query_text_embedding, inclusive_P_star.T)
        inclusive_P_star_embeddings = F.normalize(torch.tensor(inclusive_P_star_embeddings), dim=-1).numpy()

        inclusive_P_star_class_embeddings = np.matmul(class_prompt_embedding.astype(np.float32), inclusive_P_star.T)
        inclusive_P_star_class_embeddings = F.normalize(torch.tensor(inclusive_P_star_embeddings), dim=-1).numpy().astype(np.float32)
        
        def get_P0_local(input_embed):
            rewrite_pair_list = []
            for e_i in range(inclusive_candidate_prompt_embedding.shape[0]):
                for e_j in range(e_i+1, inclusive_candidate_prompt_embedding.shape[0]):
                    rewrite_pair_list.append(((inclusive_candidate_prompt_embedding[e_i] - inclusive_candidate_prompt_embedding[e_j])/2).reshape(1,-1) )


            sub_local_embeddings = np.concatenate(rewrite_pair_list)
            print(sub_local_embeddings.shape)
            sub_local_embeddings = np.concatenate([spurious_prompt_embedding, sub_local_embeddings])
            P0_local = get_proj_matrix(sub_local_embeddings)
            #
            P0_local_embeddings = np.matmul(input_embed, P0_local.T)
            P0_local_embeddings = F.normalize(torch.tensor(P0_local_embeddings), dim=-1).numpy()
            return P0_local, P0_local_embeddings
        P0_local, P0_local_embeddings = get_P0_local(query_text_embedding)
        #######

        if att_to_debias == 'gender':
            num_neighbors = 100
            K = 100
        else: 
            num_neighbors = 10
            K = 500
        ref_dataset = reference_embeddings_dataset
        spurious_class_list = ref_spurious_class_list
        
        target_dist = ref_spurious_att_prior




        bend_vlm_embeddings, x_mean, y_means = legrange_text(query_text_embedding, reference_embeddings_dataset, spurious_label=att_to_debias, 
                                                spurious_class_list=att_elements, num_neighbors=num_neighbors, proj_matrix = P0_local, normalize=normalize)



        llint_embeddings = get_debiased_embeddings_INLP(query_text_embedding, torch.tensor(reference_embeddings_dataset["embedding"]), 
                                                        ref_gender_list, P_init=P_init)
        #
        bend_vlm_class_embeddings = np.concatenate([legrange_text(class_prompt_embedding[i,:], reference_embeddings_dataset, spurious_label=att_to_debias, 
                                                spurious_class_list=att_elements, num_neighbors=num_neighbors, proj_matrix = P0_local, normalize=normalize)[0]
                                                for i in range(2)]).astype(np.float32)


        llint_class_embeddings = get_debiased_embeddings_INLP(class_prompt_embedding, torch.tensor(reference_embeddings_dataset["embedding"]), 
                                                        ref_gender_list, P_init=P_init)

        #

        result_dict[query_class]['Vanilla'][f'fold_{k_fold}'] = get_metrics(query_text_embedding, query_class, att_to_debias, 
                                                K, prior_for_metric, target_spurious_class_list, name='Vanilla', target_dataset = target_embeddings_dataset,
                                                QUERY_IS_LABELED=query_is_labeled, class_embeddings=class_prompt_embedding)


        result_dict[query_class]['P0'][f'fold_{k_fold}'] = get_metrics(P0_embeddings, query_class, att_to_debias, K, prior_for_metric, 
                                                        target_spurious_class_list, name='P0', target_dataset = target_embeddings_dataset, QUERY_IS_LABELED=query_is_labeled,
                                                        class_embeddings=P0_class_embeddings)

        result_dict[query_class]['Inclusive_P0'][f'fold_{k_fold}'] = get_metrics(inclusive_P_star_embeddings, query_class, att_to_debias, K, prior_for_metric,
                                                    target_spurious_class_list, name='Inclusive_P0', target_dataset = target_embeddings_dataset, QUERY_IS_LABELED=query_is_labeled,
                                                    class_embeddings=inclusive_P_star_class_embeddings)


        print('***'*7)
        print()
        result_dict[query_class]['bend_vlm'][f'fold_{k_fold}'] = get_metrics(bend_vlm_embeddings, query_class, att_to_debias, K, prior_for_metric, 
                                                            target_spurious_class_list, name='bend_vlm', target_dataset = target_embeddings_dataset, 
                                                            QUERY_IS_LABELED=query_is_labeled, class_embeddings=bend_vlm_class_embeddings)

        result_dict[query_class]['llint'][f'fold_{k_fold}'] = get_metrics(llint_embeddings.numpy(), query_class, att_to_debias, K, prior_for_metric, 
                                                            target_spurious_class_list, name='llint', target_dataset = target_embeddings_dataset, 
                                                            QUERY_IS_LABELED=query_is_labeled, class_embeddings=llint_class_embeddings)
        print('--'*7)
        print()
####################################################

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

####################################################

methods = ['Vanilla', 'P0', 'Inclusive_P0', 'bend_vlm', 'llint']



kl_mean = []
kl_ci =[]

max_skew_mean =[]
max_skew_ci =[]

#
auc_mean =[]
auc_ci =[]
gap_mean =[]
gap_ci =[]
#
accuracy_mean =[]
accuracy_ci =[]
#

for m in methods:
    kl_avg_list = []
    max_skew_avg_list = []

    #
    auc_avg_list = []
    gap_avg_list = []
    #
    accuracy_avg_list = []

    for fold in range(1):
        kl_list = []
        max_skew_list =[]

        #
        auc_list =[]
        gap_list =[]
        #
        accuracy_list = []

        for att in result_dict.keys():
            kl_list.append(result_dict[att][m][f"fold_{fold}"]['kl_prior'])
            max_skew_list.append(result_dict[att][m][f"fold_{fold}"]['max_skew_prior'])

            #
            auc_list.append(result_dict[att][m][f"fold_{fold}"]['worst_auc_roc_val'])
            gap_list.append(result_dict[att][m][f"fold_{fold}"]['auc_roc_gap'])
            #
            accuracy_list.append(result_dict[att][m][f"fold_{fold}"]['accuracy'])

        kl_avg_list.append(np.mean(kl_list))
        max_skew_avg_list.append(np.mean(max_skew_list))

        #
        auc_avg_list.append(np.mean(auc_list))
        gap_avg_list.append(np.mean(gap_list))
        #
        accuracy_avg_list.append(np.mean(accuracy_list))

    _m, _h = mean_confidence_interval(kl_avg_list)
    kl_mean.append(_m)
    kl_ci.append(_h)

    _m, _h = mean_confidence_interval(max_skew_avg_list)
    max_skew_mean.append(_m)
    max_skew_ci.append(_h)

    #
    _m, _h = mean_confidence_interval(auc_avg_list)
    auc_mean.append(_m)
    auc_ci.append(_h)
    _m, _h = mean_confidence_interval(gap_avg_list)
    gap_mean.append(_m)
    gap_ci.append(_h)
    #
    _m, _h = mean_confidence_interval(accuracy_avg_list)
    accuracy_mean.append(_m)
    accuracy_ci.append(_h)

df = pd.DataFrame({'Method': methods, 'KL_Div': kl_mean, 'KL_CI': kl_ci, 'MaxSkew': max_skew_mean, 
                   'MaxSkew_CI': max_skew_ci,
                   #
                #    'Worst_Group_Auc_Roc': auc_mean,
                #    'Worst_Group_Auc_Roc_CI': auc_ci,
                #    'Gap': gap_mean,
                #    'Gap_CI': gap_ci
                   #
                    'Accuracy': accuracy_mean,
                    'Accuracy_CI': accuracy_ci
                   })
df.to_csv('./out.csv')


