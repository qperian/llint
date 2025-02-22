import pandas as pd

split = pd.read_csv('../debias_vl/discriminative/datasets/data/CelebA/list_eval_partition.csv', header=0)
meta = pd.read_csv('../debias_vl/discriminative/datasets/data/CelebA/list_attr_celeba.csv', header=0)
print(split["partition"].value_counts())
train_fnames = split[split["partition"] == 0]["image_id"]
val_fnames = split[split["partition"] == 1]["image_id"]
test_fnames = split[split["partition"] == 2]["image_id"]
print(meta.iloc[: , 1:].head())
# meta_train = meta[meta["image_id"].isin(train_fnames)].to_csv('../debias_vl/discriminative/datasets/data/CelebA/meta_train.csv', index=False)
# meta_val = meta[meta["image_id"].isin(val_fnames)].to_csv('../debias_vl/discriminative/datasets/data/CelebA/meta_val.csv', index=False)
# meta_test = meta[meta["image_id"].isin(test_fnames)].to_csv('../debias_vl/discriminative/datasets/data/CelebA/meta_test.csv', index=False)
print((meta.iloc[: , 1:] > 0).astype('int').groupby(["Male"]).sum())
((meta.iloc[: , 1:] > 0).astype('int').groupby(["Male"]).sum()).to_csv('./meta_ratios.csv')