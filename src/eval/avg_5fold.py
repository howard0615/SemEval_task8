import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(args):
    score_dir = "/workplace/yhcheng/SemEval_task8/workspace/simple_transformers/sentence_classification/eval/word_format/score"
    fold1 = pd.read_csv(os.path.join(score_dir, "{}_bs{}_ga{}_lr{}".format(args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr), "fold-1.csv"))
    fold2 = pd.read_csv(os.path.join(score_dir, "{}_bs{}_ga{}_lr{}".format(args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr), "fold-2.csv"))
    fold3 = pd.read_csv(os.path.join(score_dir, "{}_bs{}_ga{}_lr{}".format(args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr), "fold-3.csv"))
    fold4 = pd.read_csv(os.path.join(score_dir, "{}_bs{}_ga{}_lr{}".format(args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr), "fold-4.csv"))
    fold5 = pd.read_csv(os.path.join(score_dir, "{}_bs{}_ga{}_lr{}".format(args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr), "fold-5.csv"))
    avg_prec, avg_recall, avg_f1 =[], [], []
    for e in range(1, args.epoch+1):
        avg_prec.append(np.average([fold1.precision[e-1], fold2.precision[e-1], fold3.precision[e-1], fold4.precision[e-1], fold5.precision[e-1]]))
        avg_recall.append(np.average([fold1.recall[e-1], fold2.recall[e-1], fold3.recall[e-1], fold4.recall[e-1], fold5.recall[e-1]]))
        avg_f1.append(np.average([fold1.f1[e-1], fold2.f1[e-1], fold3.f1[e-1], fold4.f1[e-1], fold5.f1[e-1]]))
    
    avg_df = pd.DataFrame({"epoch": [e for e in range(1, args.epoch+1)], "avg_precision": avg_prec, "avg_recall":avg_recall, "avg_f1":avg_f1})
    avg_df.to_csv(os.path.join(os.path.join(score_dir, "{}_bs{}_ga{}_lr{}".format(args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr),"avg_score.csv")), index=False)

    plt.plot(fold1.epoch.to_list(), fold1.f1.to_list(), color="green", label="fold1-f1", linestyle=":")
    plt.plot(fold2.epoch.to_list(), fold2.f1.to_list(), color="olive", label="fold2-f1", linestyle=":")
    plt.plot(fold3.epoch.to_list(), fold3.f1.to_list(), color="aquamarine", label="fold3-f1", linestyle=":")
    plt.plot(fold4.epoch.to_list(), fold4.f1.to_list(), color="orange", label="fold4-f1", linestyle=":")
    plt.plot(fold5.epoch.to_list(), fold5.f1.to_list(), color="purple", label="fold5-f1", linestyle=":")
    plt.plot(fold5.epoch.to_list(), avg_f1, color="darkblue", label="average-f1", linewidth=2.0)

    plt.legend(loc="best", shadow=True)

    plt.xlabel("epochs")
    plt.ylabel("macro-f1score")
    plt.title("{}_bs{}_ga{}_lr{} sen_cls 5-fold macro f1".format(args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr))
    plt.savefig("/workplace/yhcheng/SemEval_task8/workspace/simple_transformers/sentence_classification/eval/img/{}_bs{}_ga{}_lr{}-sen_cls-5-fold.png".format(args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr))

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
