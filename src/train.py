import argparse
import logging
import os

import pandas as pd
import torch
from simpletransformers.classification import (ClassificationArgs,
                                               ClassificationModel)

Error_log = "log/sentence_cls_5fold_finetune.log"

dev_logger = logging.basicConfig(filename=Error_log, filemode='a', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',\
                                datefmt = '%m/%d/%Y %I:%M:%S %p')



def main(args):
    logging.info(f"\n\n========= Model : [ {args.model_name} ] ===========\n\n")

    OUTPUT_FOLDER = args.output_dir

    for k in range(1, args.kfold+1):
        
        if args.train_or_not == 1:
            train_df = pd.read_csv(os.path.join(args.data_dir, f"fold-{k}", f"sentence_train_fold-{k}.csv"))
            val_df = pd.read_csv(os.path.join(args.data_dir, f"fold-{k}", f"sentence_val_fold-{k}.csv"))

            train_df = train_df.drop(
                columns=["post_id", "subreddit_id", "sentence_id"])
            val_df = val_df.drop(
                columns=["post_id", "subreddit_id", "sentence_id"])

            train_df = train_df.rename(columns={"sentence": "text"})
            val_df = val_df.rename(columns={"sentence": "text"})

            model_output_folder = OUTPUT_FOLDER+"{}_bs{}_ga{}_lr{}/fold-{}".format(args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr, k)

            # train ===========================================================================
            cuda_available = torch.cuda.is_available()
            model_args = ClassificationArgs()
            model_args.best_model_dir = model_output_folder+"best_model"
            model_args.num_train_epochs = args.epoch
            model_args.train_batch_size = args.train_batch_size
            model_args.eval_batch_size = args.val_batch_size
            model_args.evaluate_during_training = False
            model_args.learning_rate = args.lr
            model_args.logging_steps = 500
            model_args.gradient_accumulation_steps = args.grad_acc
            model_args.output_dir = model_output_folder
            model_args.save_steps = -1
            model_args.save_eval_checkpoints = True
            model_args.labels_list = ["O", "claim",
                                    "per_exp", "claim_per_exp", "question"]

            try:
                model = ClassificationModel(
                    args.model_type, args.model_name, num_labels=5, args=model_args, use_cuda=cuda_available,
                )
                logging.info(f"Success Setting up {args.model_name}")
            except Exception as e:
                logging.error("Got Error during setting up training model!!\nException:\n"+str(e))

            try:
                model.train_model(train_df=train_df,
                                show_running_loss=True)
                logging.info(f"Success training  {args.model_name}")

            except Exception as e:
                # Maybe it is CUDA out of memory
                logging.error("Got Error during training model!!!\nException:\n" + str(e))

        # val ===========================================================================


        try:
            checkpoints = os.listdir(model_output_folder)
            checkpoints = [c for c in checkpoints if "epoch" in c]

            eval_output_csv = OUTPUT_FOLDER+"{}_bs{}_ga{}_lr{}/val_csv/fold-{}/".format(args.model_name.replace("/", "-"), args.train_batch_size, args.grad_acc, args.lr, k)
            if not os.path.exists(eval_output_csv):
                os.mkdir(eval_output_csv)

            for checkpoint in checkpoints:
                logging.info(f"Running Classification model {args.model_name} with checkpoint: {checkpoint}")
                epoch = checkpoint.split("-epoch-")[1]
                cuda_available = torch.cuda.is_available()
                model_args = ClassificationArgs()
                model_args.eval_batch_size = 32
                model_args.labels_list = ["O", "claim",
                                        "per_exp", "claim_per_exp", "question"]
                
                try:
                    model = ClassificationModel(args.model_type, model_name = model_output_folder+"{}".format(checkpoint), num_labels=5, args=model_args, use_cuda=cuda_available,)
                    logging.info(f"Succes Setting up Classification model!")

                    predictions, raw_outputs = model.predict(val_df.text.to_list())

                    predict_df = val_df
                    predict_df["prediction"] = predictions
                    predict_df.to_csv(eval_output_csv+"prediction_epoch-{}.csv".format(epoch), index=False)

                except Exception as e:
                    logging.error(str(e))
        except Exception as e:
            logging.error("Running predict has got wrong!")
            logging.error(str(e))

        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_dir", type=str, default="../data/5-fold/")
    parser.add_argument("--output_dir", type=str, default="../sentence_classification_output/5-fold/")
    parser.add_argument("--train_or_not", type=int, default=1)
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--val_batch_size", type=int)
    parser.add_argument("--grad_acc", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epoch", type=int)

    args = parser.parse_args()

    main(args)
