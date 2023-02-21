import argparse
import logging
import os

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

"""
data format should be csv in SemEval task 8 test submission format
"""
# pred_path = "word_format/val_stage1_df_bi_2.csv"
# pred_path = "/workplace/yhcheng/SemEval_task8/workspace/eval/word_format/sentence_classification/bert-large-uncased.csv"

labels = ["claim", "per_exp", "claim_per_exp", "question", "O"]



def check_data_format(t_df, p_df):
    """
    column_title = ["subreddit_id", "post_id", "words", "labels"]
    """
    assert len(t_df)==len(p_df) , "Truth_csv & Predict_csv have different length truth: {} predict: {}".format(len(t_df), len(p_df))
    


def main(args):
    logging.info(f"====== Getting validation score on {args.model_name} ========")
    word_format_dir = "/workplace/yhcheng/SemEval_task8/workspace/simple_transformers/sentence_classification/eval/word_format"
    model_dir = "{}_bs{}_ga{}_lr{}".format(args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr)
    
    for f in range(1, 6):
        truth_data_dir = os.path.join(args.gold_data_dir, f"fold-{f}", f"sentence_val_fold-{f}_word_format.csv")
        eval_epoch = []
        eval_prec = []
        eval_recall = []
        eval_f1 = []
        for epoch in range(1, args.epoch+1):
            truth_df = pd.read_csv(truth_data_dir)
            pred_df = pd.read_csv(os.path.join(word_format_dir, model_dir, f"fold-{f}", "prediction_epoch-{}.csv".format(epoch)))

            check_data_format(truth_df, pred_df)

            truth_labels = truth_df.labels.to_list()
            pred_labels = pred_df.labels.to_list()
        
            # precision, recall, fscore, support = precision_recall_fscore_support(y_true=truth_labels, y_pred=pred_labels, labels=labels)
            ma_precision, ma_recall, ma_fscore, ma_support = precision_recall_fscore_support(y_true=truth_labels, y_pred=pred_labels, labels=labels, average="macro")
            logging.info("{:<8}\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t".format(epoch, ma_precision, ma_recall, ma_fscore))
            eval_epoch.append(epoch)
            eval_prec.append(ma_precision)
            eval_recall.append(ma_recall)
            eval_f1.append(ma_fscore)
        # print("Label\t\tPrecision\tRecall\t\tFScore\t\tSupport")
        # for i, label in enumerate(labels):
        #     print("{:<8}\t{:.3f}\t\t{:.3f}\t\t{:.3f}\t\t{:.3f}".format(label, precision[i], recall[i], fscore[i], support[i]))
        if not os.path.exists(os.path.join(word_format_dir, "score", model_dir)):
            os.mkdir(os.path.join(word_format_dir, "score", model_dir))
        eval_score = pd.DataFrame({"epoch":eval_epoch, "precision":eval_prec, "recall": eval_recall, "f1": eval_f1})
        eval_score.to_csv(os.path.join(word_format_dir, "score", model_dir, f"fold-{f}.csv"), index=False)
        logging.info("-----------------------------------------------------------------------------")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_data_dir",type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--val_batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--grad_acc", type=int)
    args = parser.parse_args()
    main(args)
