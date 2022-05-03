
## mDPR
This code is mostly same as the original DPR repository with some minor modifications. The code is based on [Dense Passage Retriever](https://github.com/facebookresearch/DPR) and we modify the code to support more recent version of huggingface transformers. 

### Installation
Please install the dependencies by running the command below: 

```
pip install -r requirements.txt
```

### Download models

- Baseline (1): mDPR trained on our training data (see details below) without iterative training process. 


```
mkdir models
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_shared_task_all_langs_w100.tsv
wget https://nlp.cs.washington.edu/xorqa/cora/models/mDPR_biencoder_best.cpt
unzip mGEN_model.zip
mkdir embeddings
cd embeddings
for i in 0 1 2 3;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_shared_task_embeddings/wiki_emb_en_$i 
done
for i in 0 1 2 3;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/embeddings_baseline1/wiki_emb_others_$i   
done
```

- Baseline (2)): [CORA (Asai et al., 2021)](https://github.com/AkariAsai/CORA) public model

```
mkdir models
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_shared_task_all_langs_w100.tsv
wget https://nlp.cs.washington.edu/xorqa/cora/models/mDPR_biencoder_best.cpt
unzip mGEN_model.zip
mkdir embeddings
cd embeddings
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_shared_task_embeddings/embeddings/wiki_emb_en_$i 
done
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_shared_task_embeddings/embeddings/wiki_emb_xor_$i  
done
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_shared_task_embeddings/embeddings/wiki_others_emb__$i  
done
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_shared_task_embeddings/embeddings/wiki_others_emb_ms_tr_km_$i  
done
```
### Data
#### Training data using DPR's NQ training data + XOR-TyDi QA gold paragraph data

```
wget  https://nlp.cs.washington.edu/xorqa/cora/data/base_mdpr_train_dev_data/mia2022_mdpr_train.json
wget  https://nlp.cs.washington.edu/xorqa/cora/data/base_mdpr_train_dev_data/mia2022_mdpr_xor_dev.json
```

The original data is from 
- [DPR's Natural Questions train data](https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz)
- [XOR-TyDiQA's gold paragraph data](https://nlp.cs.washington.edu/xorqa/XORQA_site/data/trans_data_all_langs.zip)

If you further augment your training data using them or the Natural Questions / TyDi QA, please make sure that you **do not** use any additional QA data from those datasets (i.e., questions whose question ids are not included in our official training data).  

#### Training data with adversarial paragraphs
Recent work has shown that using a trained DPR model to mine harder negative passages can improve retrieval performance. See detailed discussions at [the original DPR repository](https://github.com/facebookresearch/DPR#new-march-2021-retrieval-model).          

We create additional training and dev data set by augmenting positive and negative passages from the top 50 retrieval results of our mDPR models. Please see the details of this process at [create_adverarial_data.py](create_adverarial_data.py). 

```
wget  https://nlp.cs.washington.edu/xorqa/cora/data/mia_adversarial_mdpr/mia_train_adversarial.json
wget  https://nlp.cs.washington.edu/xorqa/cora/data/mia_adversarial_mdpr/mia_xor_dev_adversarial.json
```
### Training
1. Initial training 

We first train the DPR models using gold paragraph data from Natural Questions, XOR QA and TyDi QA. 

```
python -m torch.distributed.launch \
    -nproc_per_node=8 train_dense_encoder.py \
    --max_grad_norm 2.0 \
    --encoder_model_type hf_bert \
    --pretrained_model_cfg bert-base-multilingual-uncased \
    --seed 12345 --sequence_length 256 \
    --warmup_steps 300 --batch_size 16  --do_lower_case \
    --train_file /path/to/train/data \
    --dev_file /path/to/eval/data \
    --output_dir /path/to/output/dir \
    --learning_rate 2e-05 --num_train_epochs 40 \
    --dev_batch_size 6 --val_av_rank_start_epoch 30
```

2. Generate Wikipedia embeddings
After you train the DPR encoders, you need to generate Wikipedia passage embeddings. Please create a Wikipedia passage file following the instruction in the `wikipedia_preprocess` directory. The script to generate multilingual embeddings using 8 GPUs is as follows:

```sh
for i in {0..7}; do
  export CUDA_VISIBLE_DEVICES=${i}
  nohup python generate_dense_embeddings.py  --model_file /path/to/model/checkpoint --batch_size 64 --ctx_file /path/to/wikipedia/passage/file --shard_id ${i} --num_shards 8 --out_file ./embeddings_multilingual/wikipedia_split/wiki_emb > ./log/nohup.generate_wiki_emb.ser23_3_multi.${i} 2>&1 &
done
```
Note that when you generate embeddings for the 13 target languages, you may experience out of memory issue when you load the Wikipedia passage tsv file (the total wikipedia passage size is 24GB * 8 GPU). 
We recommend you to generate English embeddings first, and then do the same for the remaining languages. 

3. Retrieve Wikipedia passages for train data questions
Following prior work, we retrieve top passages for the train data questions and use them to train our generator. Once you generate train data, you can retrieve top passages by running the command below. 

```
python dense_retriever.py \
    --model_file /path/to/model/checkpoint \
    --ctx_file /path/to/wikipedia/passage/file --n-docs 100 \
    --qa_file /path/to/input/qa/file \
    --encoded_ctx_file "{glob expression for generated files}" \
    --out_file /path/to/prediction/outputs  \
    --validation_workers 4 --batch_size 64 
```

After run train your generator, please run the script to create new mDPR train data and repeat the steps from 1 using the new data. 

### Evaluations
You can run the evaluation using the same command as the step 3 in training.     
For example, to run the evaluation on the XOR QA dev data, you can run the command below.      

```
python dense_retriever.py \
    --model_file ../models/mDPR_biencoder_best.cpt \
    --ctx_file ../models/all_w100.tsv \
    --qa_file ../data/xor_dev_full_v1_1.jsonl \
    --encoded_ctx_file "../models/embeddings/wiki_emb_*" \
    --out_file xor_dev_dpr_retrieval_results.json \
    --n-docs 20 --validation_workers 1 --batch_size 256 --add_lang
```
Due to the large number of the multilingual passages embeddings, retrieving passages takes more time than English only DPR.

### Retrieved results after mDPR initial training 
The top 50 passages retrieved by mDPR after initial training for our training, development sets of MKQA and XOR-TyDi QA are available at the following locations.

- Training data
```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia_shared_training_dpr_retrieval_results.json
```

- XOR QA development data

```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia_shared_xorqa_development_dpr_retrieval_results.json
```

- MKQA development data
The retrieval results for MKQA subsets are available here:

```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_non_iterative_baselines_mkqa_dev.zip
unzip mia2022_non_iterative_baselines_mkqa_dev.zip
```

- Test data

```
wget https://nlp.cs.washington.edu/xorqa/cora/models/retriever_results_test_no_answers.zip
unzip retriever_results_test_no_answers.zip
```
