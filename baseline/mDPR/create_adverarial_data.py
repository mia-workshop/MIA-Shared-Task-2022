import json
import random
import argparse
from tqdm import tqdm
import jsonlines
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
    parser.add_argument("--qa_file", default=None, type=str)
    parser.add_argument("--orig_train_file", default=None, type=str)
    parser.add_argument("--train_retr_results", default=None, type=str)
    parser.add_argument("--output_fn", default=None, type=str)

    args = parser.parse_args()

    orig_open_data = read_jsonlines(args.qa_file)
    orig_train_data = json.load(open(args.orig_train_file))
    train_retr_results = json.load(open(args.train_retr_results))

    qid2orig_data = {item["q_id"]: item for item in orig_train_data}
    qid2retr_results = {item["q_id"]: item for item in train_retr_results}

    new_data = []
    skip = 0
    p_count = []
    n_count = []
    for item in tqdm(orig_open_data):
        qid = item["id"]
        retr_results = qid2retr_results[qid]
        positives = []
        negatives = []
        for ctx in retr_results["ctxs"]:
            if ctx["has_answer"] is True:
                positives.append(ctx)
            else:
                negatives.append(ctx)
        new_train_sample = qid2orig_data[qid] if qid in qid2orig_data else {}

        if qid not in qid2orig_data:
            new_train_sample["question"] = item["question"]
            new_train_sample["answers"] = item["answers"]
            new_train_sample["q_id"] = item["id"]
            new_train_sample["negative_ctxs"] = []
            new_train_sample["hard_negative_ctxs"] = []
            new_train_sample["positive_ctxs"] = []

        new_train_sample["positive_ctxs"] += positives
        hard_negatives_all = negatives + new_train_sample["hard_negative_ctxs"]
        sample_indices =  random.sample(range(len(hard_negatives_all)), k=min(50, len(hard_negatives_all)))
        hard_negatives = [ctx for idx, ctx in enumerate(hard_negatives_all) if idx in sample_indices]
        new_train_sample["hard_negative_ctxs"] = hard_negatives

        if len(new_train_sample["positive_ctxs"]) == 0:
            skip += 1
            continue
        else:
            p_count.append(len(new_train_sample["positive_ctxs"]))
            n_count.append(len(new_train_sample["hard_negative_ctxs"]))
            assert "question" in new_train_sample
            assert "answers" in new_train_sample
            assert "q_id" in new_train_sample
            new_data.append(new_train_sample)
    with open(args.output_fn, 'w') as outfile:
        json.dump(new_data, outfile)
    print("processed {0} examples: {1} final examples.".format(len(orig_open_data),len(orig_open_data) - skip ))
    print("avg positive ctxs number: {0}, avg negative ctxs number:{1}".format(np.mean(p_count), np.mean(n_count)))

if __name__=="__main__":
    main()