import argparse
import logging
import os

import pandas as pd


def split_text(df):
    all_post_id, all_reddit_id, all_sentence_id, all_tokens, all_labels = [], [], [], [], []
    for i, row in df.iterrows():
        tokens = row["text"].replace("\n", " ").split()
        labels = [row["prediction"] for l in range(len(tokens))]
        all_tokens.extend(tokens)
        all_labels.extend(labels)
    return pd.DataFrame({"words": all_tokens, "labels":all_labels})
    

        

def main(args):
    # output_{}_bs{}_ga{}_lr{}
    sentence_format_dir = "/workplace/yhcheng/SemEval_task8/workspace/simple_transformers/sentence_classification/5-fold"
    word_format_dir = "/workplace/yhcheng/SemEval_task8/workspace/simple_transformers/sentence_classification/eval/word_format"
    model_dir="{}_bs{}_ga{}_lr{}".format(args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr)
    model_dir = model_dir.replace("/", "-")
    sen_model_dir = os.path.join(sentence_format_dir, model_dir, "val_csv")
    for f in range(1, 6):
        # 5 fold
        fold_model_dir = os.path.join(sen_model_dir, f"fold-{f}")
        for epoch in range(1, args.epoch+1):
            checkpoint_val_prediction_df = pd.read_csv(fold_model_dir + "/prediction_epoch-{}.csv".format(epoch)) 
            checkpoint_val_prediction_df = split_text(checkpoint_val_prediction_df)
            if not os.path.exists(word_format_dir + "/" + model_dir):
                os.mkdir(word_format_dir + "/" + model_dir)
            if not os.path.exists(os.path.join(word_format_dir, model_dir, f"fold-{f}")):
                os.mkdir(os.path.join(word_format_dir, model_dir, f"fold-{f}"))
            checkpoint_val_prediction_df.to_csv(os.path.join(word_format_dir, model_dir, f"fold-{f}", "prediction_epoch-{}.csv".format(epoch)), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",type=str)
    parser.add_argument("--train_data_dir",type=str)
    parser.add_argument("--val_data_dir",type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--val_batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--grad_acc", type=int)
    args = parser.parse_args()
    main(args)
