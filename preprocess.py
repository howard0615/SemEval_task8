import argparse
import json
import os

import pandas as pd

from src.preprocess.data_utils import (mk_dir, remove_deleted_post,
                                       sentence_label)
from src.preprocess.sentence_segmentation import split_sen


def main(args):
    # print(args.train)
    if args.train==True:
        print( "train")
        assert os.path.exists(os.path.join(args.data_dir, "st1_public_data", "st1_train_inc_text.csv")), "Please put the st1_public_data folder with st1_train_inc_text.csv under args.data_dir {}".format(os.path.join(args.data_dir, "st1_public_data", "st1_train_inc_text.csv"))

        train_df = pd.read_csv(os.path.join(args.data_dir, "st1_public_data", "st1_train_inc_text.csv"))

        # remove deleted post
        train_new_df, deleted_train_df = remove_deleted_post(train_df)

        mk_dir(os.path.join(args.data_dir, "new_data_csv"))
        mk_dir(os.path.join(args.data_dir, "deleted_data_csv"))
        
        train_new_df.to_csv(os.path.join(args.data_dir, "new_data_csv", "st1_train_new.csv"))
        deleted_train_df.to_csv(os.path.join(args.data_dir, "deleted_data_csv", "st1_train_del.csv"), index=False)        

        # trankit tokenize split sentence segmentation
        mk_dir(os.path.join(args.data_dir, "trankit_token"))
        
        print("running trankit segmentation of train data")
        train_tok = split_sen(args, train_new_df)
        
        print("saved train segmentation at {}".format(os.path.join(args.data_dir, "trankit_token", "stage1_train_tokenized.json")))
        json.dump(train_tok, open(os.path.join(args.data_dir, "trankit_token", "stage1_train_tokenized.json"), "w"), ensure_ascii=False, indent=2)
        train_tok = json.load(open(os.path.join(args.data_dir, "trankit_token", "stage1_train_tokenized.json"), "r"))
        # labeling train data in sentence domain
        train_sentence_label_df = sentence_label(train_tok)
        mk_dir(os.path.join(args.data_dir, "sentence_grain"))
        train_sentence_label_df.to_csv(os.path.join(args.data_dir, "sentence_grain", "all_train_sentence_grain_label.csv"), index=False)

    
    if args.test==True:
        print("test")
        assert os.path.exists(os.path.join(args.data_dir, "st1_public_data", "st1_test_inc_text.csv")), "Please put the st1_public_data folder with st1_test_inc_text.csv under args.data_dir {}".format(os.path.join(args.data_dir, "st1_public_data", "st1_test_inc_text.csv"))

        test_df = pd.read_csv(os.path.join(args.data_dir, "st1_public_data", "st1_test_inc_text.csv")) if args.test else None
        test_new_df, deleted_test_df = remove_deleted_post(test_df)
        
        test_new_df.to_csv(os.path.join(args.data_dir, "new_data_csv", "st1_test_new.csv"), index=False)
        deleted_test_df.to_csv(os.path.join(args.data_dir, "deleted_data_csv", "st1_test_del.csv"), index=False)        
        
        print("running trankit segmentation of test data")
        test_tok = split_sen(args, test_new_df, is_test=True)
        
        print("saved test segmentation at {}".format(os.path.join(args.data_dir, "trankit_token", "stage1_test_tokenized.json")))
        json.dump(test_tok, open(os.path.join(args.data_dir, "trankit_token", "stage1_test_tokenized.json"), "w"), ensure_ascii=False, indent=2)
        test_tok = json.load(open(os.path.join(args.data_dir, "trankit_token", "stage1_test_tokenized.json"), "r"))
        # labeling test data in sentence domain
        test_sentence_label_df = sentence_label(test_tok, is_test=True)
        test_sentence_label_df.to_csv(os.path.join(args.data_dir, "sentence_grain", "all_test_sentence_grain_label.csv"), index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="home directory of data folder", default="./data")
    parser.add_argument("--train", type=bool, help="also processing train data?", default="True")
    parser.add_argument("--test", type=bool, help="also processing test data?", default="false")
    args = parser.parse_args()
    main(args)
