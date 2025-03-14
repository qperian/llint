import numpy as np
np.random.seed(42) 
fold_dict = {}
for i in range(5):
    fold_dict[i] = {}
    indices = np.arange(len(embed_list))
    np.random.shuffle(indices)

    split_point = int(len(indices) * (1 - 0.5))

    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    fold_dict[i]['train_indices'] = [int(x) for x in train_indices]
    fold_dict[i]['test_indices'] = [int(x) for x in test_indices]

with open(f'data/fold_indices/{dataset_name}_{args.name}_featurized_{_MODEL_NAME}_folds.jsonl', 'w') as fout:
    json.dump(fold_dict, fout)