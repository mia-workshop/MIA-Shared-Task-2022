import jsonlines
import json
from statistics import mean
import os
from tqdm import tqdm
from nltk.translate import bleu
import MeCab
from collections import Counter
import string
import re
import argparse
import sys
from pythainlp.tokenize import word_tokenize as th_tokenizer
from khmernltk import word_tokenize as km_tokenizer
import jieba.posseg as pseg



wakati = MeCab.Tagger("-Owakati")

lang_dic = {'telugu': 'te', 'swahili': 'sw', 'thai': 'th', 'finnish': 'fi', 'indonesian': 'id',
            'japanese': 'ja', 'russian': 'ru', 'arabic': 'ar', 'english': 'en', 'bengali': 'bn',
            "korean": "ko", "spanish": "es", "hebrew": "he", "swedish": "sv", "danish": "da", "german": "de",
            "hungarian": "hu", "italian": "it", "khmer": "km", "malay": "ms", "dutch": "nl",
            "norwegian": "no", "portuguese": "pt", "turkish": "tr", "vietnamese": "vi", "french": "fr", "polish": "pl",
            "chinese (simplified)": "zh_cn",  "chinese (hong kong)": 'zh_hk', "chinese (traditional)": "zh_tw"}

langs = ['tr', 'hu', 'zh_hk', 'nl', 'ms', 'zh_cn', 'ja', 'de', 'ru', 'pl', 'fi', 'pt', 'km',
         'it', 'fr', 'he', 'vi', 'zh_tw', 'no', 'da', 'th', 'sv', 'es', 'ar', 'en', 'ko', 'en']

def tokenize_th_text(text):
    tokens = th_tokenizer(text, engine="newmm")
    tokens = [token for token in tokens if token != " "]
    return " ".join(tokens)

def tokenize_zh_text(text):
    tokens = pseg.cut(text)
    tokens = [w.word for w in tokens]
    tokens = [token for token in tokens if token != " "]
    return " ".join(tokens)

def tokenize_km_text(text):
    tokens = km_tokenizer(text)
    tokens = [token for token in tokens if token != " "]
    return " ".join(tokens)

def read_jsonlines(eval_file_name):
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def load_tydi_answer(tydi_eval_open_domain_data):
    answer_dict = {}
    eval_data = read_jsonlines(tydi_eval_open_domain_data)
    for item in eval_data:
        answer_dict[item["id"]] = item["answers"]
    return answer_dict


def normalize_answer(s):
    # TODO: should we keep those counter removal?
    def remove_counter(text):
        return text.replace("年", "").replace("歳", "").replace("人", "").replace("년", "")

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_counter(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# 3. XOR-Full Evaluation
def calculate_f1_em_bleu(dataset, predictions):
    lang_dict = {lang: {"count": 0, "f1": 0, "bleu": 0, "em": 0}
                 for lang in lang_dic.values()}

    for qa in dataset:
        lang = qa["lang"]
        gts = qa["answers"]
        if gts[0] == "No Answer":
            continue
        q_id = qa["id"].split("_")[0]

        lang_dict[lang]["count"] += 1
        if q_id not in predictions:
            print("no answers")
            continue
        pred = predictions[q_id]
        if isinstance(gts, str):
            gts = [gts]

        final_gts = []
        # for the languages where white spaces are not widely used for word tokenization, we use the same word tokenizers on both targets and predictions and calculate word-level F1.
        if lang == "ja":
            for gt in gts:
                gt = wakati.parse(gt)
                final_gts.append(gt)
            final_pred = wakati.parse(pred.replace("・", " ").replace("、", ","))
        elif lang == "zh_cn" or  lang == "zh_hk" or lang == "zh_tw":
            for gt in gts:
                gt = tokenize_zh_text(gt)
                final_gts.append(gt)
            final_pred = tokenize_zh_text(pred)
        elif lang == "th":
            for gt in gts:
                gt = tokenize_th_text(gt)
                final_gts.append(gt)
            final_pred = tokenize_th_text(pred)
        elif lang == "km":
            for gt in gts:
                gt = tokenize_km_text(gt)
                final_gts.append(gt)
            final_pred = tokenize_km_text(pred)
        else:
            final_gts = gts
            final_pred = pred
        lang_dict[lang]["f1"] += metric_max_over_ground_truths(
            f1_score, final_pred, final_gts)
        lang_dict[lang]["bleu"] += bleu(final_gts, pred)
        lang_dict[lang]["em"] += metric_max_over_ground_truths(
            exact_match_score, final_pred, final_gts)
    # finalize scores
    for lang, scores in lang_dict.items():
        if scores["count"] == 0:
            continue
        for score_key in scores:
            if "count" != score_key:
                lang_dict[lang][score_key] = scores[score_key]/scores["count"]
    return lang_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None, type=str)
    parser.add_argument("--pred_dir",
                        default=None, type=str)
    parser.add_argument("--txt_file", action="store_true")
    parser.add_argument("--target", type=str, nargs='+')

    args = parser.parse_args()
    # load dpr results
    results_all = {}
    for lang in tqdm(langs):
        if lang not in args.target:
            continue
        dataset = read_jsonlines(os.path.join(args.data_dir, "mkqa-{}.jsonl".format(lang)))
        # fix file path
        if args.txt_file is True:
            tmp_preds = open(pred_file).read().split("\n")
            predictions = {}
            for item, pred in zip(dataset, tmp_preds):
                predictions[item["id"]] = pred
        else:
            predictions = json.load(open(os.path.join(args.pred_dir, "mkqa_final_pred_{}_cora.json".format(lang))))
        results = calculate_f1_em_bleu(dataset, predictions)
        results_all[lang] = results[lang]

    f1_total, em_total, bleu_total = 0.0, 0.0, 0.0
    total_num = 0
    lang_count = 0

    for lang in results_all:
        if results_all[lang]["count"] == 0:
            continue
        lang_count += 1
        f1_total += results_all[lang]["f1"]
        em_total += results_all[lang]["em"]
        bleu_total += results_all[lang]["bleu"]
        total_num += results_all[lang]["count"]
        print("Evaluating the performance on {0} for {1} examples".format(
            lang, results_all[lang]["count"]))
        print("F1: {0}, EM:{1}, BLEU:{2}".format(
            results_all[lang]["f1"] * 100, results_all[lang]["em"] * 100, results_all[lang]["bleu"] * 100))
    print("avg f1: {}".format(f1_total / lang_count * 100))
    print("avg em: {}".format(em_total / lang_count * 100))


if __name__ == "__main__":
    main()
