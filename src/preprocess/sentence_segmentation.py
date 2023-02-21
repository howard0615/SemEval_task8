import argparse
import ast
import json
import os

import pandas as pd
from tqdm import tqdm
from trankit import Pipeline


def split_sen(args, semeval_df: pd.DataFrame, is_test=False):
    """
    {
        "_post_id":{
            "subreddit_id": "_subreddit_id",
            "text": "_text",
            "stage1_labels": _stage1_labels(list),
            "segmentation":{
                'text': 'Hello! This is Trankit.',  # input string
                'sentences': [ # list of sentences
                    {
                    'id': 1, 'text': 'Hello!', 'dspan': (0, 6), 'tokens': [...]
                    },
                    {
                    'id': 2,  # sentence index
                    'text': 'This is Trankit.',  'dspan': (7, 23), # sentence span
                    'tokens': [ # list of tokens
                        {
                        'id': 1, # token index
                        'text': 'This', 'upos': 'PRON', 'xpos': 'DT',
                        'feats': 'Number=Sing|PronType=Dem',
                        'head': 3, 'deprel': 'nsubj', 'lemma': 'this', 'ner': 'O',
                        'dspan': (7, 11), # document-level span of the token
                        'span': (0, 4) # sentence-level span of the token
                        },
                        {'id': 2...},
                        {'id': 3...},
                        {'id': 4...}
                    ]
                    }
                ]
            }
        }
    }
    """
    pipeline = Pipeline("english", gpu=True)
    seg_dict = {}
    semeval_df["stage1_labels"] = semeval_df["stage1_labels"].apply(lambda x: ast.literal_eval(x)) if not is_test else None
    for idx, row in tqdm(semeval_df.iterrows(), total=semeval_df.shape[0], desc="Trankit sentence segmentation: "):
        csv_buf = {}
        csv_buf['subreddit_id'] = row['subreddit_id']
        csv_buf['text'] = row['text']
        csv_buf["segmentation"] = pipeline(csv_buf['text'])
        csv_buf["stage1_labels"] = row["stage1_labels"] if not is_test else None
        
        # post_id will be the index of the setence segmentation
        seg_dict[row["post_id"]] = csv_buf

    return seg_dict    
    
    