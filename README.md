# MIA 2022 Shared Task on Cross-lingual Open-Retrieval Question Answering. 
This is an official repository for MIA 2022 Shared Task on Cross-lingual Open-Retrieval Question Answering. Please refer the details in [our Shared Task call](https://mia-workshop.github.io/shared_task.html).  

Cross-lingual Open Question Answering is a challenging multilingual NLP task, where given questions are written in a user’s preferred language, a system needs to find evidence in large-scale document collections written in many different languages, and return an answer in the user’s preferred language, as indicated by their question. 

We evaluate models' performance in 14 languages, 7 of which will not be covered in our training data. 

The full list of the languages:
- Languages with training data: Arabic (`ar`), Bengali (`bn`), English (`en`), Finnish (`fi`), Japanese (`ja`), Korean (`ko`), Russian (`ru`), Telugu (`te`)
- Languages without training data: Spanish (`es`), Khmer (`km`), Malay (`ms`), Swedish (`es`), Turkish (`tr`), Chinese (simplified) (`zh-cn`)
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

*NOTE: Test data will be released in March.*
#### MIA 2022 XOR-TyDi QA data 
The data is available at [data/mia_2022_dev_xorqa.jsonl](https://github.com/mia-workshop/MIA-Shared-Task-2022/tree/main/data/eval/mia_2022_dev_xorqa.jsonl). 

The number of the examples in XOR-TyDi development and test data is as follows:

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

The data is available at [data/eval/mkqa_dev.zip]https://github.com/mia-workshop/MIA-Shared-Task-2022/blob/main/data/eval/mkqa_dev.zip). 

```
cd data/eval
unzip mkqa_dev.zip
```
MKQA development has 1,758 questions per language, and 5,000 questions per language. All of the questions are parallel across different languages. 

Each file contains questions for each target language.


### Training Data
#### Constrained Setup
Our training data for the **constrained** setup consists of English open-QA data from Natural Questions-open ([Kwiatkowski et al., 2019](https://research.google/pubs/pub47761/); [Lee et al., 2019](https://arxiv.org/abs/1906.00300)) and the XOR-TyDi QA train data. 

The training is available at [data/train/mia_2022_train_data.jsonl.zip](https://github.com/mia-workshop/MIA-Shared-Task-2022/blob/main/data/train/mia_2022_train_data.jsonl.zip). 
We encourage participants to do data augmentation using Machine Translation or structured knowledge sources (e.g., Wikidata, Wikipedia interlanguage links), and as long as you do not use additional human annotated QA data or data, you submissions will be considered as constrained setup.

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

We also release the training data for our DPR-based baseline, which is created by collecting training for Natural Questions available at [DPR repo](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz) and [XOR-TyDI gold paragraph data](https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xorqa_reading_comprehension_format.zip).  Please see the details in the `baseline` section. 

#### Unconstrained Setup
For the **unconstrained** setup, you may use additional human-annotated question answering data. Yet, you must not use additional data from Natural Questions or XOR-TyDi QA for training. 
Please note that participants using additional human-annotated question-answer data must clarify it and provide the details of the additional resources used during the training. 



## Evaluate

Participants will run their systems on the evaluation files (without answer data) and then submit their predictions to our competition site hosted at eval.ai. Systems will first be evaluated using automatic metrics: **Exact match (EM)** and **token-level F1**. Although EM is often used as a primarily evaluate metric for English open-retrieval QA, the risk of surface-level mismatching (Min et al., 2021) can be more pervasive in cross-lingual open-retrieval QA. Therefore, we will use F1 as our primary metric and rank systems using their macro averaged F1 scores.

Due to the difference of the datasets' nature, we will calculate macro-average scores on XOR-TyDi and MKQA datasets, and then take the average of the XOR-TyDi QA average {F1, EM} and MKQA average {F1, EM}.

### Dependencies
For non-spacing languages (i.e., Japanese, Khmer and Chinese) we use token-level tokenizers, Mecab, khmernltk and jieba to tokenize both predictions and ground-truth answers.

Please install the required dependencies by running the command below.

```
cd eval_scripts
pip install -r requirements.txt
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
The baseline codes are available at [baseline](baseline). 
Our baseline model is the state-of-the-art [CORA](https://github.com/AkariAsai/CORA), which runs a multilingual DPR model to retrieve documents from many different languages and then generate the final answers in the target languages using a multilingual seq2seq generation models. We have two versions:
1. **CORA with iterative training**: We run the publicly available CORA's trained models on our evaluation set. We generate dense embeddings for all of the target languages using their mDPR bi encoders as some of the languages (e.g., Chinese - simplified) are not covered by the CORA's original embeddings. The original COPRA models are trained via their new iterative training framework. We also release the retrieval results of this models [here](). 
2. **CORA without iterative training**: We train mDPR and mGEN without iterative training process. 
3. **BPR (Yamada et al., 2021)**: When we increases the retrieval target to more languages, the inference latency and storage requirements increases quickly. We will plan to introduce a new baseline using Binary Passage Retriever (**BP**R; [Yamada et al., 2021](https://arxiv.org/abs/2106.00882)) as a memory efficient baseline. 

We also release translation results for the evaluation dataset by MT models for MKQA and XOR-TyDi QA evaluation set. 

### Results of Baselines 
The results on the development set of Baseline (1) are below. We will add the results of other baselines shortly. The predictions results are available at [MIA2022_sample_predictions](https://drive.google.com/drive/folders/11SewNZ8v_KV4lEE3zFVpHkkBHyuMTI5W?usp=sharing). 

We also release the mDPR retrieval results for dev and test set, and will release the mDPR results for the training set upon request. 
You can download the mDPR results from: 
```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_mDPR_results_xor_mkqa_dev.zip
```

#### Final results 

- XOR QA (sample prediction file: [`xor_dev_output.json`](https://drive.google.com/file/d/18VKYStO8s0_bW-R-gzdHbCBdOEVdRJIX/view?usp=sharing))


| Language | F1 | EM |
| :-----: | :-------:| :------: |
| Arabic (`ar`) | 51.3 |  36.0 |
| Bengali (`bn`) | 28.7 | 20.2 |
| Finnish (`fi`) | 44.4 | 35.7 |
| Japanese (`ja` )| 43.2 | 32.2 |
| Korean (`ko`) |  29.8 | 23.7 |
| Russian (`ru`) |  40.7 | 31.9 |
| Telugu (`te`) |  40.2 | 32.1 |

- MKQA (sample prediction file: [`mkqa_dev_output.zip`](https://drive.google.com/file/d/1P1VHyQgdzQW4EeQtwvmy34luwB6xDq06/view?usp=sharing))


| Language | F1 | EM |
| :-----: | :-------:| :------: |
| Arabic (`ar`) | 8.8  | 5.7 |
| English (`en`) | 27.9 | 24.5 |
| Spanish (`es`) | 24.9 | 20.9 |
| Finnish (`fi`) | 23.3 | 20.0 |
| Japanese (`ja`) | 15.2 | 6.3 |
| Khmer (`km`) |  5.7 | 4.9 |
| Korean (`ko`) |  8.3 | 6.3 |
| Malaysian (`ms`) | 22.6  |19.7 |
| Russian (`ru`) | 14.0  | 9.4 |
| Swedish (`sv`) |  24.1 | 21.1|
| Turkish (`tr`) |  20.6 | 16.7 |
| Chinese-simplified (`zh_cn`) | 13.1  | 6.1 |

## Submission
Our shared task will be hosted at [eval.ai](https://eval.ai/). Submission details will be available early March with test dataset.

## Contact
If you have any questions, please feel free to email (`akari[at]cs.washington.edu`) or start a Github issue with a mention to `@AkariAsai]` or `@shayne-longpre`