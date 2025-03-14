import torch
import numpy as np
import pandas as pd
import scipy.stats
from transformers import CLIPModel, AutoProcessor
import torch.nn.functional as F
import yaml
import json
from datasets import load_dataset, DatasetDict
from datasets import concatenate_datasets
from bend_utils import *
from evaluate import evaluate
import queries

from Rotational import rotational_equalize
from LLINT import full_LLINT_debias, get_INLP_P
from SFID import get_SFID_indices, get_SFID_embeddings

######
# Set run parameters
######
#config = yaml.safe_load(open("experimental_configs/celeba_various_gender_clip-vit-base-patch16.yml"))
# config = yaml.safe_load(open("experimental_configs/fairface_stereotype_race_clip-vit-base-patch16.yml"))
config = yaml.safe_load(open("experimental_configs/utk_gender.yml"))
#config = yaml.safe_load(open("experimental_configs/celeba_stereotype_gender_clip-vit-large-patch14.yml"))
RUN_NAME = 'sped_up'#used for naming output files
FOLDS = 1 #how many folds to perform cross validation over (must be <= # when creating dataset)

print(config)
############################################
# Load parameters from config file, and set up general parameters
#-----
    # config file gives broad details (each property below)
    # queries.py is the names of the specific queries (e.g., burgular, evil person, etc. for "stereotype")
    # specific prompts are in query_templates folder
############################################
_MODEL_NAME = config['model_ID'].split('/')[-1]
att_to_debias = config['att_to_debias']
ref_dataset_name = config['ref_dataset_name']
target_dataset_name = config['target_dataset_name']
model_ID = config['model_ID'] 
query_type = config['query_type'] #e.g. 'stereptype', specifies which set of prompts to test on
random_seed = config['random_seed'] 

if query_type == 'hair':
    query_classes = queries.hair_classes
elif query_type == 'stereotype':
    query_classes = queries.stereotype_classes
elif query_type == 'various':
    query_classes = queries.various_classes
else:
    print(f'{query_type} not implemented')

print(f"query_classes: {query_classes}")

#
normalize = True
lam = 1000
#

if att_to_debias == 'race':
    if ref_dataset_name[:7] == 'UTKFace':
        att_elements = ['White', 'Black', 'Asian', 'Indian', 'Latino Hispanic']
    else:
        att_elements = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']
elif att_to_debias == 'gender':
    att_elements = ['Male', 'Female']
else:
    print('{att_to_debias} not implemented')


if query_type in ('hair', 'various'):
    query_is_labeled = True
else:
    query_is_labeled = False

if att_to_debias == 'gender':
    num_neighbors = 100
    if query_is_labeled:
        K = 100
    else:
        K = 500
else: 
    num_neighbors = 10
    K = 500
############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############################################
# Set up CLIP model and processor function to turn text into embeddings
############################################

model_id = config['model_ID']

vl_model = CLIPModel.from_pretrained(model_ID).to(device)
processor = AutoProcessor.from_pretrained(model_ID)

####################################################
# Define functions used to load saved datasets into Huggingface datasets objects
####################################################
def utk_ethnicity_map(x):
    ethnicity_map = {
        0: 'White',
        1: 'Black', 
        2: 'Asian',
        3: 'Indian',
        4: 'Latino Hispanic'
    }
    return ethnicity_map[x]



def load_embedding_dataset(ds_path):
    if ds_path.split('/')[-1] == 'celeba_large_train_featurized_clip-vit-large-patch14.jsonl':
        # embeddings_dataset_half1 = load_dataset("json", data_files='data/half1_'+ds_path.split('/')[-1], split='train')
        # embeddings_dataset_half2 = load_dataset("json", data_files='data/half2_'+ds_path.split('/')[-1], split='train')
        # dd_to_concat = [embeddings_dataset_half1, embeddings_dataset_half2]
        # embeddings_dataset = concatenate_datasets(dd_to_concat)
        data_files = {"train": ['data/half1_'+ds_path.split('/')[-1], 'data/half2_'+ds_path.split('/')[-1]]}
        embeddings_dataset = load_dataset("json", data_files=data_files, split='train')
    else:
        embeddings_dataset = load_dataset("json", data_files=ds_path, split='train')
    embeddings_dataset = embeddings_dataset.with_format("np", columns=["embedding"], output_all_columns=True)

    if 'Male' in embeddings_dataset.features.keys():
        embeddings_dataset = embeddings_dataset.map(
            lambda x: {"gender": 'Male' if x['Male'] == 1 else 'Female'})
    if 'utk_race' in embeddings_dataset.features.keys():
        embeddings_dataset = embeddings_dataset.map(
            lambda x: {"race": utk_ethnicity_map(x['utk_race'])})
    if 'UTK' in ds_path:
        embeddings_dataset = embeddings_dataset.map(
            lambda x: {"gender": 'Male' if x['gender'] == 0 else 'Female'})
        embeddings_dataset = embeddings_dataset.map(
            lambda x: {"race": utk_ethnicity_map(x['race'])})
        

    if normalize:
        embeddings_dataset = embeddings_dataset.map(
            lambda x: {"embedding": x['embedding'].reshape(-1)/np.linalg.norm(x['embedding'].reshape(-1))})
    else:
        embeddings_dataset = embeddings_dataset.map(
            lambda x: {"embedding": x['embedding'].reshape(-1)})
    return embeddings_dataset

####################################################
# Load train and test datasets
####################################################
train_embeddings_dataset = load_embedding_dataset(f'data/{ref_dataset_name}_featurized_{_MODEL_NAME}.jsonl')
_embeddings_dataset = load_embedding_dataset(f'data/{target_dataset_name}_featurized_{_MODEL_NAME}.jsonl') # this is the test dataset

# Load query texts / prompts for each query type
with open(f"query_templates/{query_type}_{att_to_debias}_query_templates.json", 'r') as file:
    instantiated_search_classes = json.load(file)

# load fold indices which split up datasets into subsets for cross validation 
# (and split up train/test if target and test datasets are same)
with open(f"data/fold_indices/{target_dataset_name}_featurized_{_MODEL_NAME}_folds.jsonl", 'r') as file:
    fold_dict = json.load(file)


reference_embeddings_dataset = train_embeddings_dataset#.select(fold_dict[str(0)]['train_indices'])
target_embeddings_dataset = _embeddings_dataset#.select(fold_dict[str(0)]['test_indices'])

####################################################

ref_spurious_class_list = att_elements
target_spurious_class_list = att_elements#['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']#att_elements

gendercode = {"Male": 1, "Female": 0}
ref_gender_list = np.array(list(map(lambda g: gendercode[g], reference_embeddings_dataset["gender"])))

att_codes = {att_elements[i] : i for i in range(len(att_elements))}#{att_elements[i] : i for i in range(len(att_elements))}

spurious_prompt = ["A photo of a man", "A photo of a woman"]
spurious_prompt_embedding = get_embeddings(spurious_prompt, vl_model, processor, normalize).to('cpu').numpy()
P0 = get_proj_matrix(spurious_prompt_embedding).astype(np.float32)
#P_init = get_INLP_P(F.normalize(torch.tensor(reference_embeddings_dataset["embedding"])), ref_att_list)

result_dict = {}

P_inits = {}

sfid_important_indices, sfid_mean_features_lowconfidence, low_conf_images = get_SFID_indices(train_embeddings_dataset, device, att_to_debias, ref_dataset_name, _MODEL_NAME)

for k_fold in range(FOLDS):

    if target_dataset_name == ref_dataset_name:
        reference_embeddings_dataset = train_embeddings_dataset.select(fold_dict[str(k_fold)]['train_indices'])
    else:
        reference_embeddings_dataset = train_embeddings_dataset
        if k_fold > 0:
            P_inits[k_fold] =P_inits[k_fold-1]
            continue

    target_embeddings_dataset = _embeddings_dataset.select(fold_dict[str(k_fold)]['test_indices'])

    ref_att_list = np.array(list(map(lambda g: att_codes[g], reference_embeddings_dataset[att_to_debias])))
    P_inits[k_fold] = get_INLP_P(F.normalize(torch.tensor(reference_embeddings_dataset["embedding"])), ref_att_list)


####################################################
for query_class in query_classes:
    result_dict[query_class] = {}


    if query_is_labeled:
        query_class_prompt = instantiated_search_classes[query_class]['classes']
        class_prompt_embedding = get_embeddings(query_class_prompt, vl_model, processor, normalize).to('cpu').numpy()
    else:
        class_prompt_embedding = None

    for k_fold in range(FOLDS):

        if target_dataset_name == ref_dataset_name:
            reference_embeddings_dataset = train_embeddings_dataset.select(fold_dict[str(k_fold)]['train_indices'])
        else:
            reference_embeddings_dataset = train_embeddings_dataset
        target_embeddings_dataset = _embeddings_dataset.select(fold_dict[str(k_fold)]['test_indices'])
        ref_att_list = np.array(list(map(lambda g: att_codes[g], reference_embeddings_dataset[att_to_debias])))

        gendercode = {"Male": 1, "Female": 0}
        ref_gender_list = np.array(list(map(lambda g: gendercode[g], reference_embeddings_dataset["gender"])))

        P_init, seps_init = P_inits[k_fold]

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
            print(f'conditional spurious att array: {spurious_att_array[eg_array==1].shape[0]}, {conditional_spurious_att_array[conditional_spurious_att_array==spurious_att].shape[0]},{spurious_att_array.shape[0]}')
            print(f'conditional spurious att prior: {conditional_spurious_att_prior}')

        if query_is_labeled:
            prior_for_metric =spurious_att_prior
        else:
            prior_for_metric = spurious_att_prior




        query_text_embedding = get_embeddings(query_text, vl_model, processor, normalize).to('cpu').numpy()
        spurious_prompt_embedding = get_embeddings(spurious_prompt, vl_model, processor, normalize).to('cpu').numpy()
        inclusive_candidate_prompt_embedding = get_embeddings(inclusive_candidate_prompt, vl_model, processor, normalize).to('cpu').numpy()

        P0 = get_proj_matrix(spurious_prompt_embedding).astype(np.float32)

        M = get_M(inclusive_candidate_prompt_embedding, S)
        G = lam * M + np.eye(M.shape[0])
        inclusive_P_star = np.matmul(P0, np.linalg.inv(G))


        P0_embeddings = np.matmul(query_text_embedding, P0.T)
        P0_embeddings = F.normalize(torch.tensor(P0_embeddings), dim=-1).numpy()

        if query_is_labeled:
            P0_class_embeddings = np.matmul(class_prompt_embedding.astype(np.float32), P0.T)
            P0_class_embeddings = F.normalize(torch.tensor(P0_class_embeddings), dim=-1).numpy().astype(np.float32)
        else:
            P0_class_embeddings = None

        inclusive_P_star_embeddings = np.matmul(query_text_embedding, inclusive_P_star.T)
        inclusive_P_star_embeddings = F.normalize(torch.tensor(inclusive_P_star_embeddings), dim=-1).numpy()

        if query_is_labeled:
            inclusive_P_star_class_embeddings = np.matmul(class_prompt_embedding.astype(np.float32), inclusive_P_star.T)
            inclusive_P_star_class_embeddings = F.normalize(torch.tensor(inclusive_P_star_embeddings), dim=-1).numpy().astype(np.float32)
        else:
            inclusive_P_star_class_embeddings = None
        
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

        ref_dataset = reference_embeddings_dataset
        spurious_class_list = ref_spurious_class_list
        
        target_dist = ref_spurious_att_prior


        SFID_query_embeddings = get_SFID_embeddings(torch.tensor(query_text_embedding), sfid_mean_features_lowconfidence, sfid_important_indices, device).detach().cpu().numpy()
        SFID_target_embeddings = get_SFID_embeddings(torch.tensor(target_embeddings_dataset["embedding"]), sfid_mean_features_lowconfidence, sfid_important_indices, device).detach().cpu().numpy()
        
        #target_embeddings_dataset.ma(lambda x: {"embedding": get_SFID_embeddings(x, sfid_mean_features_lowconfidence, sfid_important_indices, device).detach().cpu().numpy()})
        bend_vlm_embeddings, x_mean, y_means = legrange_text(query_text_embedding, reference_embeddings_dataset, spurious_label=att_to_debias, 
                                                spurious_class_list=att_elements, num_neighbors=num_neighbors, proj_matrix = P0_local, normalize=normalize)



        llint_embeddings, llint_seps, Ps_LLINT = full_LLINT_debias(query_text_embedding, torch.tensor(reference_embeddings_dataset["embedding"]), 
                                                        ref_att_list, P_init=P_init, seps_init=seps_init)
        if query_is_labeled:
            bend_vlm_class_embeddings = np.concatenate([legrange_text(class_prompt_embedding[i,:], reference_embeddings_dataset, spurious_label=att_to_debias, 
                                                    spurious_class_list=att_elements, num_neighbors=num_neighbors, proj_matrix = P0_local, normalize=normalize)[0]
                                                    for i in range(2)]).astype(np.float32)


            llint_class_embeddings, llint_class_seps, Ps_class_LLINT = full_LLINT_debias(class_prompt_embedding, torch.tensor(reference_embeddings_dataset["embedding"]), 
                                                            ref_att_list, P_init=P_init, seps_init=seps_init)
            inlp_class_embeddings = np.matmul(class_prompt_embedding,P_init.T).numpy()
        else:
            bend_vlm_class_embeddings = None
            llint_class_embeddings = None
            inlp_class_embeddings = None

        '''
        # #
        # torch.save(llint_embeddings, f'./results/{query_class}_{RUN_NAME}_ref({ref_dataset_name})_targ({target_dataset_name})_query({query_type})_att({att_to_debias})_model({_MODEL_NAME})_fold({k_fold})_llint_embeddings.pt')
        llint_seps = llint_seps[0]
        print(len(llint_seps))
        llint_combined = np.concatenate([np.vstack(seps).astype(np.float32) for seps in llint_seps])
        print([seps[1].shape for seps in llint_seps])
        print("RANK", np.linalg.matrix_rank(llint_combined))

        
        # labels = get_cos_neighbors(query_text_embedding, target_embeddings_dataset, k = 100)[1][att_to_debias]
        # neighbors =  get_cos_neighbors(query_text_embedding, target_embeddings_dataset, k = 100)[1]
        # llint_embeddings = torch.tensor(rotational_equalize(query_text_embedding, neighbors["embedding"], llint_combined, low_conf_images, labels))
        # llint_ref_images = torch.tensor(rotational_equalize(torch.tensor(target_embeddings_dataset["embedding"]), train_embeddings_dataset["embedding"], llint_combined, low_conf_images)).numpy()
        # llint_seps = llint_seps[0]
        # print(len(llint_seps))
        # llint_combined = np.concatenate([np.vstack(seps.reshape(1,-1)).astype(np.float32) for seps in llint_seps])
        print("RANK", np.linalg.matrix_rank(llint_combined))
        print("LLINT SEPS", llint_combined.shape)
        # print(util.cos_sim(llint_seps[0],llint_seps[1]))
        query_sims = util.cos_sim(query_text_embedding, llint_combined)
        print("SEP SIMS QUERY", query_sims)
        gender_sims = util.cos_sim(spurious_prompt_embedding[0]-spurious_prompt_embedding[1], llint_combined)
        print("SEP SIMS GENDERS", gender_sims)
        print(llint_combined.shape)
        print(np.abs(query_sims[0]) < 0.2)
        filtered_seps = llint_combined[np.abs(query_sims[0]) < 0.2]
        print("FILTERED SEPS", filtered_seps.shape)
        filtered_proj = get_proj_matrix(filtered_seps)
        print("RANK", np.linalg.matrix_rank(filtered_proj))

        # def get_retr_acc(query):
        #     _scores, _samples = get_cos_neighbors(query, target_embeddings_dataset, k = K)
        #     _, tot_acc = group_accuracy(_samples, query_class,  spurious_label=att_to_debias, spurious_class_list=target_spurious_class_list)
        #     return tot_acc
        
        # ret_accs = np.array([get_retr_acc(np.array(sep)) for sep in llint_combined])
        # print("RETR ACC", [get_retr_acc(np.array(sep)) for sep in llint_combined])

        filtered_seps = llint_combined#[np.abs(query_sims[0]) < 2]
        # print("FIlTErED OUT", llint_combined[np.abs(query_sims[0]) > 0.2].shape)
        # print("FILTERED SEPS", filtered_seps.shape)
        filtered_proj = get_proj_matrix(filtered_seps)
        # print("RANK", np.linalg.matrix_rank(filtered_proj))
        
        # llint_embeddings = torch.from_numpy(np.matmul(query_text_embedding, filtered_proj.T))
        '''
        debiased_embeddings = { #query, target, class
            'Vanilla': (query_text_embedding, None, class_prompt_embedding),
            'P0': (P0_embeddings, None, P0_class_embeddings),
            'Inclusive_P0': (inclusive_P_star_embeddings, None, inclusive_P_star_class_embeddings),
            'SFID': (query_text_embedding, SFID_target_embeddings.astype(np.float32), class_prompt_embedding),
            'bend_vlm': (bend_vlm_embeddings, None, bend_vlm_class_embeddings),
            'llint': (llint_embeddings.numpy(), None, llint_class_embeddings),
            'inlp': (np.matmul(query_text_embedding,P_init).numpy(), None, inlp_class_embeddings),
        }

        
        for method in debiased_embeddings.keys(): 
            query_prompt, target_set, class_prompt = debiased_embeddings[method]
            if method not in result_dict[query_class].keys():
                result_dict[query_class][method] = {}
            result_dict[query_class][method][f'fold_{k_fold}'] = get_metrics(query_prompt, query_class, att_to_debias, 
                                                K, prior_for_metric, target_spurious_class_list,
                                                name=method, target_dataset = target_embeddings_dataset,
                                                embedds=target_set,
                                                fold=k_fold, QUERY_IS_LABELED=query_is_labeled,
                                                class_embeddings=class_prompt,
                                                folder_name=f'{RUN_NAME}_ref({ref_dataset_name})_targ({target_dataset_name})_query({query_type})_att({att_to_debias})_model({_MODEL_NAME}')
            print()
####################################################

methods = debiased_embeddings.keys()

df_atts = evaluate(result_dict, methods, query_is_labeled, folds=FOLDS)

df = pd.DataFrame(df_atts)
df.to_csv(f'./results/{RUN_NAME}_ref({ref_dataset_name})_targ({target_dataset_name})_query({query_type})_att({att_to_debias})_model({_MODEL_NAME}).csv')