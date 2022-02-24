# MIA 2022 Shared Task on Cross-lingual Open-Retrieval Question Answering. 
This is an official repository for MIA 2022 Shared Task on Cross-lingual Open-Retrieval Question Answering. Please refer the details in [our Shared Task call](https://mia-workshop.github.io/shared_task.html).  

Cross-lingual Open Question Answering is a challenging multilingual NLP task, where given questions are written in a user’s preferred language, a system needs to find evidence in large-scale document collections written in many different languages, and return an answer in the user’s preferred language, as indicated by their question. 

We evaluate models' performance in 14 languages, 7 of which will not be covered in our training data. 

The full list of the languages:
- Languages with training data: Arabic, Bengali, English Finnish, Japanese, Korean, Russian, Telugu
- Languages without training data: Spanish, Khmer, Malay, Swedish, Turkish, Chinese (simplified)

### Quick Links

- [Datasets](#datasets)
- [Download](#download)
- [Evaluate](#evaluate)
- [Baseline](#baseline)
- [Submission](#submission)

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
Note that the part of the XOR-TyDi data only includes English answers due to the limitation of annotation resources ([Asai et al., 2021](https://nlp.cs.washington.edu/xorqa/)). Those questions are marked as `has_eng_answer_only: True` 

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
The shared task valuation data is originally from [XOR-TyDi QA](https://nlp.cs.washington.edu/xorqa/)([Asai et al., 2021](https://arxiv.org/abs/2010.11856)) 
and [MKQA](https://github.com/apple/ml-mkqa)[(Longpre et al., 2021)](https://arxiv.org/abs/2007.15207). 

For this shared task, we re-split development and test data, and conduct additional answer normalizations, so the number on our shared task data is not directly comparable to the results on those datasets in prior paper. 
- The number of the examples in XOR-TyDi development and test data is as follows:

| Language | dev | test |
| :-----: | :-------:| :------: |
| `ar` |  590 | 1387 |
| `bn` |  203 | 490 |
| `fi` |  1368 | 974 |
| `ja` |  1056 | 693 |
| `ko` |  1048 | 473 |
| `ru` |  910 | 1018 |
| `te` |  873 | 564 |

- MKQA development has 1,758 questions per language, and 5,000 questions per language. All of the questions are parallel across different languages. 


### Training Data
#### Constrained Setup
Our training data for the **constrained** setup consists of English open-QA data from Natural Questions-open ([Kwiatkowski et al., 2019](https://research.google/pubs/pub47761/); [Lee et al., 2019](https://arxiv.org/abs/1906.00300)) and the XOR-TyDi QA train data. 

The training data can be downloaded at [this link](https://drive.google.com/file/d/16XSf26qS0pFPpNSRarhymGbjNYKQMwvy/view?usp=sharing). 

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

We also release the training data for our DPR-based baseline, which is created by collecting training for Natural Questions available at [DPR repo](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz) and [XOR-TyDI gold paragraph data](https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xorqa_reading_comprehension_format.zip).  Please see the details in the `baseline` section. 

#### Unconstrained Setup
For the **unconstrained** setup, you may use additional human-annotated question answering data. Yet, you must not use additional data from Natural Questions or XOR-TyDi QA for training. 
Please note that participants using additional human-annotated question-answer data must clarify it and provide the details of the additional resources used during the training. 


## Download
You can download the training and evaluation data by running the command below. The test data will be release late March. 

- Training data 
```
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/mia_2022_train_data.jsonl
```
- Evaluation data (XOR-TyDi) 
```
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/mia_2022_eval_data_dev.jsonl
```

- Evaluation data (MKQA) 
```
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/mia_2022_eval_mkqa.zip
```
## Evaluate

Participants will run their systems on the evaluation files (without answer data) and then submit their predictions to our competition site hosted at eval.ai. Systems will first be evaluated using automatic metrics: **Exact match (EM)** and **token-level F1**. Although EM is often used as a primarily evaluate metric for English open-retrieval QA, the risk of surface-level mismatching (Min et al., 2021) can be more pervasive in cross-lingual open-retrieval QA. Therefore, we will use F1 as our primary metric and rank systems using their macro averaged F1 scores.

Due to the difference of the datasets' nature, we will calculate macro-average scores on XOR-TyDi and MKQA datasets, and then take the average of the XOR-TyDi QA average {F1, EM} and MKQA average {F1, EM}.

### Dependencies
For non-spacing languages (i.e., Japanese, Khmer and Chinese) we use token-level tokenizers, Mecab, khmernltk and jieba to tokenize both predictions and ground-truth answers.

Please install the following libraries to run the evaluation scripts:
```
pip install jieba
pip install khmernltk
pip install pythainlp
pip install mecab-python3
```

Please use python 3.x to run the evaluation scripts.

### Prediction file format
Your prediction file is a `json` file including a dictionary where the keys are corresponding to the question ids and the values are the answers. 

```
{"7931051574381133444": "1954年から1955年", "-6802534628745605728": "2,718", ... }:
```

### Evaluation Scripts
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
The baseline codes are available at [baseline](baseline)
Our baseline model is the state-of-the-art [CORA](https://github.com/AkariAsai/CORA), which runs a multilingual DPR model to retrieve documents from many different languages and then generate the final answers in the target languages using a multilingual seq2seq generation models. 

For those who want to focus on single component of the task, we release the retrieved results from mDPR. 

We also release translation results for the evaluation dataset by MT models for MKQA and XOR-TyDi QA evaluation set. 

## Submission
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

```
{"xor-tydi: {...}, "mkqa_ar": {...}, "mkqa_ja": {...}}
```
Once you create your submission files, please go to [our competition website]() at eval.ai. 

## Contact
If you have any questions, please feel free to email (`akari[at]cs.washington.edu`) or start a Github issue with a mention to `@AkariAsai]` or `@shayne-longpre`