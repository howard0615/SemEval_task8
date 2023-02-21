import argparse
import logging
import os

import pandas as pd
import torch
from simpletransformers.classification import (ClassificationArgs,
                                               ClassificationModel)


def main(args):
    try:
        for k in range(1, args.kfold+1):

            OUTPUT_FOLDER = args.output_dir
            val_df = pd.read_csv(os.path.join(
                args.data_dir, f"fold-{k}", f"sentence_val_fold-{k}.csv"))
            val_df = val_df.drop(
                columns=["post_id", "subreddit_id", "sentence_id"])
            val_df = val_df.rename(columns={"sentence": "text"})
            model_output_folder = OUTPUT_FOLDER+"{}_bs{}_ga{}_lr{}/fold-{}".format(
                args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr, k)

            checkpoints = os.listdir(model_output_folder)
            checkpoints = [c for c in checkpoints if "epoch" in c]

            eval_output_csv = OUTPUT_FOLDER+"{}_bs{}_ga{}_lr{}/val_csv/fold-{}/".format(
                args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr, k)

            if not os.path.exists(OUTPUT_FOLDER+"{}_bs{}_ga{}_lr{}/val_csv".format(args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr)):
                os.mkdir(OUTPUT_FOLDER+"{}_bs{}_ga{}_lr{}/val_csv".format(
                    args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr))

            if not os.path.exists(eval_output_csv):
                os.mkdir(eval_output_csv)

            for checkpoint in checkpoints:
                logging.info(
                    f"Running Classification model {args.model_name} with checkpoint: {checkpoint}")
                epoch = checkpoint.split("-epoch-")[1]
                cuda_available = torch.cuda.is_available()
                model_args = ClassificationArgs()
                model_args.eval_batch_size = 32
                model_args.labels_list = ["O", "claim",
                                          "per_exp", "claim_per_exp", "question"]

                try:
                    model = ClassificationModel(args.model_type, model_name=model_output_folder+"/{}".format(
                        checkpoint), num_labels=5, args=model_args, use_cuda=cuda_available,)
                    logging.info(f"Succes Setting up Classification model!")

                    predictions, raw_outputs = model.predict(
                        val_df.text.to_list())

                    predict_df = val_df
                    predict_df["prediction"] = predictions
                    predict_df.to_csv(
                        eval_output_csv+"prediction_epoch-{}.csv".format(epoch), index=False)

                except Exception as e:
                    logging.error(str(e))
    except Exception as e:
        logging.error("Running predict has got wrong!")
        logging.error(str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--train_or_not", type=int, default=1)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--val_batch_size", type=int)
    parser.add_argument("--grad_acc", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epoch", type=int)

    args = parser.parse_args()

    main(args)
