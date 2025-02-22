import pandas as pd

utk_meta = pd.read_csv('./data_wrangling/utk_meta.csv', header=0, sep=",")
utk_meta["age"] = utk_meta["file"].apply(lambda x: x.split("_")[0])
utk_meta["gender"] = utk_meta["file"].apply(lambda x: x.split("_")[1])
utk_meta["race"] = utk_meta["file"].apply(lambda x: x.split("_")[2])
print(len(utk_meta))
#utk_meta.to_csv('./utk_info.csv', index=False)