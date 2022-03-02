# Baselines for MIA 2022 Shared Task

Our primary baseline model is the state-of-the-art [CORA](https://github.com/AkariAsai/CORA), which runs a multilingual DPR model to retrieve documents from many different languages and then generate the final answers in the target languages using a multilingual seq2seq generation models. We have two versions:
1. **CORA with iterative training**: We run the publicly available CORA's trained models on our evaluation set. We generate dense embeddings for all of the target languages using their mDPR bi encoders as some of the languages (e.g., Chinese - simplified) are not covered by the CORA's original embeddings. The original COPRA models are trained via their new iterative training framework. 
2. **CORA without iterative training**: We train mDPR and mGEN without iterative training process. The trained models will be released shortly,

When we increases the retrieval target to more languages, the inference latency and storage requirements increases quickly. We plan to introduce a new baseline using Binary Passage Retriever (**BP**R; [Yamada et al., 2021](https://arxiv.org/abs/2106.00882)) as a memory efficient baseline, based on the official implementation available at [soseki](https://github.com/studio-ousia/soseki).

## Baseline 1: Quick evaluation
To reproduce the main results of the baseline 1, please run the following script.

```
bash run_evaluation.sh
```
## Baseline predictions

### mDPR retrieval results
To encourage those who are more interested in improving answer generation / reader components after retrieval, we release the retrieval results for the MKQA and XOR TyDi QA data. All of the retrieval results for the development set can be downloaded by running the command below. 

```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_mDPR_results_xor_mkqa_dev.zip
```

If you need mDPR results for the training set as well. please contact akari[at]cs.washington.edu. 

Retrieval results for the test set will be available once the official test data is released. 
### Final MGEN results 
All of the final prediction results of Baseline 1 are available at [MIA2022_sample_predictions](https://drive.google.com/drive/folders/11SewNZ8v_KV4lEE3zFVpHkkBHyuMTI5W?usp=sharing). 