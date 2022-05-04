# MIA 2022 Shared Task on Cross-lingual Open-Retrieval Question Answering. 
This is an official repository for MIA 2022 Shared Task on Cross-lingual Open-Retrieval Question Answering. Please refer the details in [our Shared Task call](https://mia-workshop.github.io/shared_task.html).  

**If you are interested in participating, please sign up at [this form](https://forms.gle/ioWDn4UCKyftTVCk6) to get invitations for our google group!**

### Updates
- **March 6,  2022**: We released baseline models, train and development data.
- **March 19,  2022**: We added new baseline (mDPR + mGEN trained on our official data) with prediction results. 
- **March 21,  2022**: We released test data files at [data/eval/mia_2022_test_xorqa_without_answers.jsonl (XOR test data without answer data)](data/eval/mia_2022_test_xorqa_without_answers.jsonl) and [data/eval/mkqa_test_without_answers.zip (MKQA test data without answer data)](data/eval/mkqa_test_without_answers.zip).
- **March 24,  2022**: [Our submission site](https://eval.ai/web/challenges/challenge-page/1638/my-submission) on EvalAI is now up! You can now test your predictions on the dev set, and submissions on the test set will be open on April 1. Link to the eval.ai competition site 
- **May 3,  2022**: We've released the test data in two surprise languages, **Tamil and Tagalog**. Please download the data at `eval/data` directory. [Tagalog data](data/eval/mia2022_test_surprise_tagalog_without_answers.jsonl) | [Tamil data](data/eval/mia2022_test_surprise_tamil_without_answers.jsonl). 

### Overview
Cross-lingual Open Question Answering is a challenging multilingual NLP task, where given questions are written in a user’s preferred language, a system needs to find evidence in large-scale document collections written in many different languages, and return an answer in the user's preferred language, as indicated by their question. 

We have awards + prizes for the best **Unconstrained** system, **Constrained** system, and **Creativity** awards for participants without massive compute/resources but still obtain interesting results!

We evaluate models' performance in ~~14~~16 languages, ~~7~~9 of which will not be covered in our training data. 7 languages have development data, while 2 surprise languages released two weeks before submission deadline does not have any development data. 

The full list of the languages:
- Languages with training data: Arabic (`ar`), Bengali (`bn`), English (`en`), Finnish (`fi`), Japanese (`ja`), Korean (`ko`), Russian (`ru`), Telugu (`te`)
- Languages without training data: Spanish (`es`), Khmer (`km`), Malay (`ms`), Swedish (`es`), Turkish (`tr`), Chinese (simplified) (`zh-cn`)
- *New!!* Surprise languages: Tagalog (`tl`), Tamil (`ta`)


### Quick Links

- [Datasets & Track Rules](#datasets)
- [Evaluate](#evaluate)
- [Baseline](#baseline)
- [Submission](#submission)
- [Shared Task Awards](#shared-task-awards)

## Datasets

### Data Format
Our train and evaluation data files are `jsonlines`, each of which contains a list of a json data with `id`, `question`, `lang`, `answers`, `split`, `source` (i.e., `nq` for the data from Natural Questions, `xor-tydi` for the data from XOR-TyDi QA and `mkqa` for the data from MKQA).  

e.g., 
- Natural Questions training data 
```
{
    'id': '-6802534628745605728',
    'question': 'total number of death row inmates in the us',
    'lang': 'en',
    'answers': ['2,718'],
    'split': 'train',
    'source': 'nq'
}
```

- XOR-TyDi QA training data (with answers in the target language)
```
{
    'id': '7931051574381133444', 
    'question': '『指輪物語』はいつ出版された', 
    'answers': ['1954年から1955年'], 
    'lang': 'ja', 
    'split': 'train', 
    'source': 'xor-tydi'
}
```

- XOR-TyDi QA training data (without answers in the target language)
Note that the part of the XOR-TyDi data only includes English answers due to the limitation of annotation resources ([Asai et al., 2021](https://nlp.cs.washington.edu/xorqa/)). Those questions are marked as `has_eng_answer_only: True`.

```
{
    'id': '7458071188830898854', 
    'question': 'チョンタル語はいつごろ誕生した', 
    'answers': ['5,000 years ago'], 
    'lang': 'ja', 
    'split': 'train', 
    'source': 'xor-tydi',
    has_eng_answer_only: True
}
```
### Evaluation Data
The shared task valuation data is originally from [XOR-TyDi QA](https://nlp.cs.washington.edu/xorqa/) ([Asai et al., 2021](https://arxiv.org/abs/2010.11856)) and [MKQA](https://github.com/apple/ml-mkqa) [(Longpre et al., 2021)](https://arxiv.org/abs/2007.15207). 
​
For this shared task, we re-split development and test data, and conduct additional answer normalizations, so the number on our shared task data is not directly comparable to the results on those datasets in prior paper. 

*NOTE: Test data will be released in March.*
#### MIA 2022 XOR-TyDi QA data 
The data is available at [data/mia_2022_dev_xorqa.jsonl](https://github.com/mia-workshop/MIA-Shared-Task-2022/tree/main/data/eval/mia_2022_dev_xorqa.jsonl). 

Dataset size statistics:

| Language | dev | test |
| :-----: | :-------:| :------: |
| `ar` |  590 | 1387 |
| `bn` |  203 | 490 |
| `fi` |  1368 | 974 |
| `ja` |  1056 | 693 |
| `ko` |  1048 | 473 |
| `ru` |  910 | 1018 |
| `te` |  873 | 564 |

#### MIA 2022 Shared Task MKQA data

The data is available at [data/eval/mkqa_dev.zip](https://github.com/mia-workshop/MIA-Shared-Task-2022/blob/main/data/eval/mkqa_dev.zip). 

```
cd data/eval
unzip mkqa_dev.zip
```

The MKQA dev set has 1,758 questions per language, and the test set has 5,000 questions per language. All of the questions are parallel across different languages. 

### Training Data
#### (1) Constrained Setup
Our training data for the **constrained** setup consists of English open-QA data from Natural Questions-open ([Kwiatkowski et al., 2019](https://research.google/pubs/pub47761/); [Lee et al., 2019](https://arxiv.org/abs/1906.00300)) and the XOR-TyDi QA train data. 

The training is available at [data/train/mia_2022_train_data.jsonl.zip](https://github.com/mia-workshop/MIA-Shared-Task-2022/blob/main/data/train/mia_2022_train_data.jsonl.zip). 
We encourage participants to do data augmentation using Machine Translation or structured knowledge sources (e.g., Wikidata, Wikipedia interlanguage links), and as long as you do not use additional human annotated QA data or data, you submissions will be considered as constrained setup.
NB: Using external blackbox APIs such as Google Search API / Google Translate API are not permitted in the **constrained setup** for *inference*, but they are permitted for offline data augmentation / training.

```
cd data/train
unzip mia_2022_train_data.jsonl.zip
```

| Dataset | Language | # of Examples |
| :-----: | :-------:| :------: |
| Natural Questions | `en` | 76635 |
| XOR-TyDi QA |  `ar` | 18402 |
| XOR-TyDi QA |  `bn` | 5007 |
| XOR-TyDi QA |  `fi` | 9768 |
| XOR-TyDi QA |  `ja` | 7815 |
| XOR-TyDi QA |  `ko` | 4319 |
| XOR-TyDi QA |  `ru` | 9290 |
| XOR-TyDi QA |  `te` | 6759 |

We also release the training data for our DPR-based baseline, which is created by collecting training for Natural Questions available at [DPR repo](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz) and [XOR-TyDI gold paragraph data](https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xorqa_reading_comprehension_format.zip). 
The data can be downloaded at [mia2022_shared_task_train_dpr_train.json](https://drive.google.com/file/d/1BZJ2ibAKdm-4xLddh_4D4P1e_G4x3Ogr/view?usp=sharing).
If we want to download data programmatically from Google Drive, you can use [gdown](https://github.com/wkentaro/gdown)

```py
import gdown
url = url = "https://drive.google.com/uc?id=1BZJ2ibAKdm-4xLddh_4D4P1e_G4x3Ogr"
gdown.download(url)
```
Please see more details in the `baseline` section. 


#### (2) Unconstrained Setup
For the **unconstrained** setup, you may use additional human-annotated question answering data. Yet, you must not use additional data from Natural Questions or XOR-TyDi QA for training. 
Participants using additional human-annotated question-answer data must report this and provide details of the additional resources used for training. 
NB: Using external blackbox APIs such as Google Search API / Google Translate API is permitted in the **unconstrained setup**. 


### Wikipedia Dumps
Following the original XOR-TyDi QA datasets, our baselines use the Wikipedia dump from February 2019. For the languages whose 2019 February dump is no longer available, we use October 2021 data.

#### Link to the Wikipedia dumps
You can find the links of web archive version of Wikipedia dumps at the TyDi QA repository below:
[TyDi QA's source data list](https://github.com/google-research-datasets/tydiqa/blob/master/README.md#source-data)


For the languages that are not listed here, here are the links:

Swedish Wikipedia
Spanish Wikipedia
Chinese (simplified Wikipedia
Malay Wikipedia
Khmer Wikipedia
Turkish Wikipedia


#### Preprocessed data in the DPR (100-token) format
You can download preprocessed text data where we split each article in all target languages into 100 token chuncks and concatenate all of them. The download links are below:

```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_shared_task_all_langs_w100.tsv
````
We will add this instruction in our README. Section for your feedback! Let us know if you have any more questions. The code used to preprocess Wikipedia is at baseline/wikipedia_preprocess.

## Evaluate
Participants will run their systems on the evaluation files (without answer data) and then submit their predictions to our competition site hosted at eval.ai. Systems will first be evaluated using automatic metrics: **Exact match (EM)** and **token-level F1**. Although EM is often used as a primarily evaluate metric for English open-retrieval QA, the risk of surface-level mismatching (Min et al., 2021) can be more pervasive in cross-lingual open-retrieval QA. Therefore, we will use F1 as our primary metric and rank systems using their macro averaged F1 scores.

**Final Evaluation Procedure**: Due to the difference of the datasets' nature, we will calculate macro-average scores on XOR-TyDi and MKQA datasets, and then take the average of the XOR-TyDi QA average {F1, EM} and MKQA average {F1, EM}.

### Dependencies
For non-spacing languages (i.e., Japanese, Khmer and Chinese) we use token-level tokenizers, Mecab, khmernltk and jieba to tokenize both predictions and ground-truth answers.

Please install the dependencies by running the command below before running the evaluation scripts:

```
cd eval_scripts
pip install -r requirements.txt
```

Please use python 3.x to run the evaluation scripts.

### Prediction file format
#### Evaluate locally
Your prediction file is a `json` file including a dictionary where the keys are corresponding to the question ids and the values are the answers. 

```
{"7931051574381133444": "1954年から1955年", "-6802534628745605728": "2,718", ... }:
```
#### Final submission
To submit your prediction to our eval.ai leaderboard, you have to create a single `json` file that includes predictions for all of the XOR-TyDi and MKQA subsets. Please see the details in the [submission](#submission) section. 

### Evaluate predictions locally
Please run the command below to evaluate your models' performance on MKQA and XOR-TyDi QA. 

```
python eval_xor.py --data_file /path/to/your/data/file --pred_file /path/to/prediction/file
```

For MKQA, you can run the command above for each language or you can run the command below that takes directory names of the prediction files and input data files.

```
python eval_mkqa_all.py --data_dir /path/to/data/dir --pred_dir /path/to/pred/files/dir 
```
You can limit the target languages by setting the `--target` option. You can add the target languages' language codes (e.g., `--target en es sw`)

## Baseline
The baseline codes are available at [baseline](baseline). 
Our baseline model is the state-of-the-art [CORA](https://github.com/AkariAsai/CORA), which runs a multilingual DPR model to retrieve documents from many different languages and then generate the final answers in the target languages using a multilingual seq2seq generation models. We have two versions:

1. **multilingual DPR + multilingual seq2seq (CORA without iterative training)**: We train mDPR and mGEN without iterative training process. We first train mDPR, retrieve top passages using the trained mDPR, and then fine-tuned mGEN after we preprocess and augment NQ data using WikiData as in the original CORA paper. Due to the computational costs, we do not re-train the CORA with iterative training on the new data and languages. 

2. **CORA with iterative training**: We run the publicly available CORA's trained models on our evaluation set. We generate dense embeddings for all of the target languages using their mDPR bi encoders as some of the languages (e.g., Chinese - simplified) are not covered by the CORA's original embeddings. There might be some minor differences in data preprocessing of the original CORA paper and our new data. 

### Prediction results of the baselines
We've released the final prediction results as well as the intermediate retrieval results for train, dev and test sets. 

To download the data, follow the instructions at [baseline README](baseline#intermediate-results----mdpr-retrieval-results). 
#### Final Prediction results
- Baseline 1:[MIA2022_Baseline 1 sample_predictions](https://drive.google.com/drive/folders/14Xv6enk7j4d3QKTNbB5jGjaColNffwW_?usp=sharing). 
- Baseline 2: [MIA2022_Baseline 2 sample_predictions](https://drive.google.com/drive/folders/1ePQjLOWUNiF5mr6leAhw8OG-o1h55i75?usp=sharing). 

#### Intermediate Retrieval Results
See the Instructions at [the Baseline's README](https://github.com/mia-workshop/MIA-Shared-Task-2022/tree/main/baseline#intermediate-results----mdpr-retrieval-results). 

#### Final results F1 | EM |
The final results of Baselines 2 and 3 are shown below. The final macro average scores of those baselines are: 
- Baseline 1 = `(38.9 + 18.1 ) / 2`= **28.5** 
- Baseline 2 = `(39.8 + 17.4) / 2`= **28.6** 

- XOR QA 

| Language | (2) F1 | (2) EM |  (1) F1 | (1) EM |
| :-----: | :-------:| :------: |  :-------:| :------: |
| Arabic (`ar`) | 51.3 |  36.0 | 49.7 |  33.7 |
| Bengali (`bn`) | 28.7 | 20.2 | 29.2 | 21.2 |
| Finnish (`fi`) | 44.4 | 35.7 | 42.7 | 32.9  |
| Japanese (`ja` )| 43.2 | 32.2 | 41.2 | 29.6 |
| Korean (`ko`) |  29.8 | 23.7 | 30.6 | 24.5 |
| Russian (`ru`) |  40.7 | 31.9 | 40.2 | 31.1 |
| Telugu (`te`) |  40.2 | 32.1 | 38.6 | 30.7 |
| Macro-Average |  39.8 | 30.3 | 38.9| 29.1 |

- MKQA

| Language | (2) F1 | (2) EM | (1) F1 | (1) EM |
| :-----: | :-------:| :------: | :-------:| :------: |
| Arabic (`ar`) | 8.8  | 5.7 | 8.9 | 5.1 |
| English (`en`) | 27.9 | 24.5 | 33.9 | 24.9 |
| Spanish (`es`) | 24.9 | 20.9 | 25.1 | 19.3 |
| Finnish (`fi`) | 23.3 | 20.0 | 21.1 | 17.4 |
| Japanese (`ja`) | 15.2 | 6.3 | 15.3 | 5.8 |
| Khmer (`km`) |  5.7 | 4.9 | 6.0 | 4.7 |
| Korean (`ko`) |  8.3 | 6.3 |  6.7 | 4.7 |
| Malaysian (`ms`) | 22.6  |19.7 | 24.6 | 19.7 |
| Russian (`ru`) | 14.0  | 9.4 |  15.6  | 10.6 |
| Swedish (`sv`) |  24.1 | 21.1| 25.5 | 20.6|
| Turkish (`tr`) |  20.6 | 16.7 | 20.4 |  16.1 |
| Chinese-simplified (`zh_cn`) | 13.1  | 6.1 | 13.7  | 5.7 |
| Macro-Average  | 17.4 | 13.5 |  18.1  | 12.9 |

## Submission
Our shared task is hosted at [eval.ai](https://eval.ai/web/challenges/challenge-page/1638/overview). 

### Submission file format
To be considered for the prizes, you have to submit predictions for all of the target languages included in XOR-TYDi and MKQA. A valid submission data is a dictionary with the following keys and the corresponding prediction results in the format discussed in the [Evaluation](#evaluate) section.

- `xor-tydi`
- `mkqa-ar` 
- `mkqa-en`
- `mkqa-es`
- `mkqa-fi`
- `mkqa-ja`
- `mkqa-km`
- `mkqa-ko`
- `mkqa-ms`
- `mkqa-ru`
- `mkqa-sv`
- `mkqa-tr`
- `mkqa-zh_cn`
- `sup_ta`
- `sup_tl`
​
```
{"xor-tydi: {...}, "mkqa-ar": {...}, "mkqa-ja": {...}}
```

We've released our baseline systems' prediction results in the submission format at [submission.json (dev data predictions)](sample_predictions/submission.json) and [submission_test.json (test data predictions)](sample_predictions/submission_test.json).  

## Shared Task Awards

There are 3 awards in this shared task, each with Google Cloud credits as prizes.

1. The Best **unconstrained** system obtaining the highest macro-average F1 scores.
2. The Best **constrained** system obtaining the highest macro-average F1 scores.
3. **Creativity award(s)**: We plan to give additional award(s) to systems that employ a creative approach to the problem, or undertake interesting experiments to better understand the problem. This award is designed to encourage interesting contributions even from teams without access to the largest models or computational resources. Examples include attempts to improve generalization ability/language equality, reducing model sizes, or understanding the weaknesses in existing systems.

## Contact
If you have any questions, please feel free to email (`akari[at]cs.washington.edu`) or start a Github issue with a mention to `@AkariAsai` or `@shayne-longpre`
