# Baselines for MIA 2022 Shared Task

Our primary baseline model is the state-of-the-art [CORA](https://github.com/AkariAsai/CORA), which runs a multilingual DPR (mDPR) model to retrieve documents from many different languages and then generate the final answers in the target languages using a multilingual seq2seq generation models (mGEN). 

We have two versions:
1. **(Baseline 1) mDPR + mGEN (CORA w/o iterative training)**: We train mDPR and mGEN following the procedures in the CORA paper using the MIA2022 shared task official training and development data. We do not conduct iterative training.      
The experimental results can be reproduced by `run_evaluation.sh`. This is our primary baseline. 

2. **(Baseline 2) CORA (trained models)**:
We run the models available at the CORA library on our evaluation data. For the languages that are not originally covered by the CORA repository, we run mDPR to generate passage embeddings.      
THe experimental results can be reproduced by `run_evaluation_cora.sh`.

## Quick evaluation
To reproduce the main results of the baseline 1, please run the following script.

```
bash run_evaluation.sh
```
## Baseline predictions

### Intermediate results -- mDPR retrieval results 

To encourage those who are more interested in improving answer generation / reader components after retrieval, we release the retrieval results for the MKQA and XOR TyDi QA data. All of the retrieval results for the training and development set can be downloaded by running the command below. 

- Training data

```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia_shared_training_dpr_retrieval_results.json
```

- XOR QA Development data
```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia_shared_xorqa_development_dpr_retrieval_results.json
```


- MKQA development data
```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_non_iterative_baselines_mkqa_dev.zip
unzip mia2022_non_iterative_baselines_mkqa_dev.zip
```

- Test data
```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_non_iterative_baselines_mkqa_dev.zip
unzip mia2022_non_iterative_baselines_mkqa_dev.zip
```

Retrieval results for the test set will be available once the official test data is released. 

### Final prediction results
You can download final predictions from the following links. 

- Baseline 1:[MIA2022_Baseline 1 sample_predictions](https://drive.google.com/drive/folders/14Xv6enk7j4d3QKTNbB5jGjaColNffwW_?usp=sharing). 

- Baseline 2: [MIA2022_Baseline 2 sample_predictions](https://drive.google.com/drive/folders/1ePQjLOWUNiF5mr6leAhw8OG-o1h55i75?usp=sharing). 
