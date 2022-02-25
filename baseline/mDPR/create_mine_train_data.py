import json
import random
import argparse
import csv
import os
from tqdm import tqdm
import jsonlines
import ast
import re
import numpy as np
from collections import Counter

def read_jsonlines(eval_file_name):
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open_input_fn", default=None, type=str)
    parser.add_argument("--gold_data", default=None, type=str)
    parser.add_argument("--pred_result", default=None, type=str)
    parser.add_argument("--output_fn", default=None, type=str)

    args = parser.parse_args()

    orig_open_data = read_jsonlines(args.input_fn)
    qid2lang = {item["id"]:item["lang"] for item in orig_open_data}

    # load the original gold data; see the details of the input files in README.md.  
    tsv_file = open(args.gold_para_data)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    orig_gold_data = []
    for row in read_tsv:
        orig_id = row[2].split("_")[0]
        lang = qid2lang[orig_id]
        orig_gold_data.append(
            {"id": orig_id, "lang": lang, "answers": ast.literal_eval(row[1]), \
                "title": row[3], "context":row[4], "question": row[5], "orig_id": orig_id})

    # Load predictions
    if args.pred_result.endswith(".txt"):
        preds = open(args.pred_result).read().split("\n")
    elif args.pred_result.endswith(".json"):
        preds_orig = json.load(open(args.pred_result))
        preds = []
        for data in orig_gold_data:
            pred = preds_orig[data["id"]]
            pred.append(pred)
    else:
        raise NotImplementedError

    assert len(orig_gold_data) == len(preds)

    match_para_ids = {}
    for pred, data in tqdm(zip(preds, orig_gold_data)):
        orig_id = data["id"]
        match_para_ids.setdefault(orig_id, {"positive_ctxs": [], "negative_ctxs": [], "hard_negative_ctxs": [], 'matched_ctxs': [], "question": "{0} [{1}]".format(data["question"], data["lang"]), "lang": data["lang"], "answers": data["answers"] })
        if pred in data["answers"]:
            ctx = data["context"]
            match_para_ids[orig_id]["positive_ctxs"].append(
                {"text": ctx, "title": data["title"]})
        else:
            ctx = data["context"]
            if data["answers"][0] not in ctx:
                match_para_ids[orig_id]["hard_negative_ctxs"].append(
                    {"text": ctx, "title": data["title"]})
            else:
                match_para_ids[orig_id]["matched_ctxs"].append(
                    {"text": ctx, "title": data["title"]})

    dpr_data = []
    new_positive_ids = []
    for q_id, item in match_para_ids.items():
        if len(item["positive_ctxs"]) > 0:
            item["q_id"] = q_id
            new_positive_ids.append(q_id)
        if len(item["positive_ctxs"]) == 0 and len(item["matched_ctxs"]) > 0:
            item["positive_ctxs"].append(item["matched_ctxs"][0])

        elif len(item["positive_ctxs"]) == 0 and len(item["matched_ctxs"]) == 0:
            print("examples are skipped")
            continue
        dpr_data.append(item)   

    print(dpr_data[-1])
    print(len(dpr_data))
    print(len(new_positive_ids))
    print("positive para num:{}".format(np.mean([len(item["positive_ctxs"]) for item in dpr_data])))
    print("negative para num:{}".format(np.mean([len(item["hard_negative_ctxs"]) for item in dpr_data])))

    with open(args.output_fn, 'w') as outfile:
        json.dump(dpr_data, outfile)


if __name__=="__main__":
    main()