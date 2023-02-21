import argparse
import os

import pandas as pd
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=666)

def main(args):
    data = pd.read_csv(os.path.join(args.data_dir, "sentence_grain", "all_train_sentence_grain_label.csv"))
    if not os.path.exists(os.path.join(args.data_dir, "5-fold")):
        os.mkdir(os.path.join(args.data_dir, "5-fold"))
        
    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        print(f"Fold {i}==============")
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        if not os.path.exists(os.path.join(args.data_dir, "5-fold", f"fold-{i+1}")):
            os.mkdir(os.path.join(args.data_dir, "5-fold", f"fold-{i+1}"))
        train.to_csv(args.data_dir + "/5-fold/fold-{}/sentence_train_fold-{}.csv".format(i+1, i+1), index=False)
        test.to_csv(args.data_dir + "/5-fold/fold-{}/sentence_val_fold-{}.csv".format(i+1, i+1), index=False)
        print(train.labels.value_counts())
        print(test.labels.value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="home directory of data folder", default="./data")
    args = parser.parse_args()
    main(args)
