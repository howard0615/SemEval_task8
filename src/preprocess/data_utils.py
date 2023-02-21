import os

import pandas as pd


def remove_deleted_post(df: pd.DataFrame):
    del_df = df.loc[df["text"].str.contains("deleted")]
    else_df = pd.merge(df, del_df, indicator=True, how="outer").query('_merge=="left_only"').drop('_merge', axis=1)
    return else_df, del_df


def mk_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def check_span(dspan_head, dspan_tail, value):
    for entity in value["stage1_labels"][0]["crowd-entity-annotation"]["entities"]:
        if entity["startOffset"] <= dspan_head and  dspan_tail <= entity["endOffset"]:
            """
            dspan:       -----===========--------
            entity_span  ---===============------ V
            """
            return entity["label"]
        elif entity["startOffset"] <= dspan_head and entity["endOffset"] <= dspan_tail and dspan_head < entity["endOffset"]:
            """
            dspan:       -----===========--------
            entity_span  ---===========---------- V
            """
            dspan_len = dspan_tail - dspan_head
            entity_in_dspan_len = entity["endOffset"]-dspan_head
            if entity_in_dspan_len >= dspan_len/2:
                # entity span 佔dspan 裡的一半以上，就會被認定為有這label
                return entity["label"]
            elif entity_in_dspan_len < dspan_len/2:
                # 若沒有，就會先跳過，並不會直接回傳"O"，因可能他主要佔的是其他的entity
                pass
        elif dspan_head <= entity["startOffset"] and dspan_tail <= entity["endOffset"] and entity["startOffset"] < dspan_tail:
            """
            dspan:       -----===========--------
            entity_span  ------------======------ V
            """
            dspan_len = dspan_tail - dspan_head
            entity_in_dspan_len = dspan_tail - entity["endOffset"]
            if entity_in_dspan_len >= dspan_len/2:
                return entity["label"]
            elif entity_in_dspan_len < dspan_len/2:
                pass
        elif dspan_head <= entity["startOffset"] and entity["endOffset"] <= dspan_tail:
            """
            dspan:       -----===========--------
            entity_span  --------======---------- V
            """
            return entity["label"]
    # no any entity where on the dspan ; return "O"
    return "O"

def sentence_label(tok: dict, is_test=False):
    post_id, subreddit_id, sentence_id, sentences, labels = [], [], [], [], []

    if not is_test:
        for key, value in tok.items():
            for sen in value["segmentation"]["sentences"]:
                dspan_head, dspan_tail = sen["dspan"][0], sen["dspan"][1]
                """
                dspan:       -----===========--------
                entity_span  ---===============------ V
                entity_span  ---===========---------- V
                entity_span  ------------======------ V
                entity_span  --------======---------- V
                entity_span  ------------------====-- X
                """
                this_sen_label = check_span(dspan_head, dspan_tail, value)

                post_id.append(key)
                subreddit_id.append(value["subreddit_id"])
                sentence_id.append(int(sen["id"])-1)
                sentences.append(sen["text"].replace("\n", " "))
                labels.append(this_sen_label)
    
        df = pd.DataFrame({"post_id": post_id, "subreddit_id": subreddit_id, "sentence_id": sentence_id, "sentences": sentences, "labels": labels})
        return df              

    if is_test:
        post_id, subreddit_id, sentence_id, sentences = [], [], [], []

        for key, value in tok.items():
            for sen in value["segmentation"]["sentences"]:
                post_id.append(key)
                subreddit_id.append(value["subreddit_id"])
                sentence_id.append(int(sen["id"])-1)
                sentences.append(sen["text"].replace("\n", " "))
        df = pd.DataFrame({"post_id": post_id, "subreddit_id": subreddit_id, "sentence_id": sentence_id, "sentences": sentences})

        return df
