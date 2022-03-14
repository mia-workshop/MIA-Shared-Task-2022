import jsonlines
from collections import Counter
from tqdm import tqdm
import wptools
import json
import argparse
import random


def read_jsonlines(eval_file_name):
    lines = []
    print("loading examples from {0}".format(eval_file_name))
    with jsonlines.open(eval_file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def wikidata_alignment(answer):
    page = wptools.page(answer)
    answer_dict = {}
    try:
        page.get_more()
        for item in page.data["languages"]:
            answer_dict[item["lang"]] = item["title"]
        return answer_dict
    except:
        print("cannot find the answer")
        return None


def postprocess(answer_string):
    if "(" in answer_string:
        return answer_string.split("(")[0]
    else:
        return answer_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fp", default=None, type=str)
    parser.add_argument("--dpr_data", action="store_true")
    parser.add_argument("--output_fp", default=None, type=str)
    parser.add_argument("--sample_num", default=None, type=int)

    args = parser.parse_args()

    if args.dpr_data is True:
        # read input from DPR format file to align the English gold articles to the corresponding ones in the other languages.
        input_data = json.load(open(args.input_fp))
    else:
        # read input data in the xor qa format.
        input_data = read_jsonlines(args.input_fp)
        if args.sample_num is not None:
            input_data = random.sample(input_data, k=args.sample_num)
    output_data = {}
    print("original input data num:{}".format(len(input_data)))

    for idx, item in tqdm(enumerate(input_data)):
        if args.dpr_data is True:
            answers = [item["positive_ctxs"][0]["title"]]
            q_id = idx
        else:
            answers = item["answers"]
            q_id = item["id"]

        # remove this all digit cases?
        for answer in list(set(answers)):
            if str(answer).isdigit() == False:
                translated_answers = wikidata_alignment(answer)
                if translated_answers is not None:
                    output_data.setdefault(q_id, {})
                    for lang, answer in translated_answers.items():
                        translated_answer = postprocess(answer)
                        output_data[q_id][lang] = translated_answer
            else:
                translated_answer = str(answer)
                output_data.setdefault(q_id, {})
                output_data[q_id]["numeric"] = translated_answer
    print("found aligned answers for {} questions".format(len(output_data)))

    print("final data num: {}".format(len(output_data)))
    with open(args.output_fp, 'w') as outfile:
        json.dump(output_data, outfile)


if __name__ == "__main__":
    main()
