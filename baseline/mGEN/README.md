## mGEN
This directory contains the code for the mGEN component. The code is originally based on [the transformers' implementation of RAG](https://github.com/huggingface/transformers/tree/v4.2.1/examples/research_projects/rag).

### Installation
Please download the dependencies by running the command below:

```
pip install -r requirements.txt
```

### Data
To lift the burden of training and running computationally expensive retrieval models, we release the retrieval results (top 50 passages) for the training, dev and test set (will be released when the official test data will be released).     
You can also use retrieval results of your own retriever(s).

Here, we first download the mDPR retrieval results for the official training and dev sets and convert the data format.

#### mDPR retrieval results

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

#### Data format

Our fine-tuning logic is based on scripts from [`examples/seq2seq`](https://github.com/huggingface/transformers/tree/master/examples/seq2seq). We accept training data in the same format as specified there - we expect a directory consisting of 6 text files: 

```bash
train.source
train.target
val.source
val.target
test.source
test.target
```
Each line contains each source/target sentence. 

#### Convert mDPR output to mGEN train data format
This scripts convert the DPR output file into mGEN train data format. Please set the file names for train, dev and test data (`--train_fp`, `--dev_fp`, and `--test_fp`) and the output directory name (`--output_dir`). You can choose the number of the top DPR retrieved passages (`--top_n`). 

```
python3 convert_dpr_retrieval_results_to_seq2seq.py \
    --train_fp /path/to/dpr/output/train/data.json --dev_fp /path/to/dpr/output/dev/data.json  \
    --output_dir /path/to/mgen/data/dir \
    --top_n 15 --add_lang
```

- Augment training data with WikiData
CORA introduces a WikiData-based simple data augmentation approach for languages not covered in the human annotated training data.       
In particular, this approach retrieves Wikipedia entities in many languages corresponding to the original English answers in Natural Questions, and automatically generate cross-lingual mGEN training data by replacing the English answers with the target languages and appending language tags to the questions.      

Please follow the steps below if you want to try this data augmentation approach. 

1. Retrieve corresponding Wikipedia entities 
```
python align_wikidata.py --input_fp /path/to/input/qa/data.jsonl --output_fp /path/to/output/entity/file/name.json --sample_num 10000
```
The API can get slow, so for our baselines, we sample 10k NQ questions and retrieve corresponding entities. We obtained entities for about 6k questions. 

2. Augment data with Wikipedia entity file

```
python convert_dpr_retrieval_results_to_seq2seq.py  \
    --train_fp /path/to/dpr/output/train/data.json \
    --dev_fp /path/to/dpr/output/dev/data.json \
    --output_dir /path/to/mgen/data/dir \
    --top_n 15 --add_lang \
    --ent_fp /path/to/output/entity/file/name.json
```

You can download the processed mGEN training data:
```
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia_shared_task_mgen_nq_added.zip
```
### Training
Please specify the `model_type`, `model_name_or_path` and `gpus` (the number of GPUs to be used during fine-tuning).

- Train `mt5-base` based model

```sh
python finetune_mgen.py \
    --data_dir /path/to/your/data/dir \
    --output_dir /path/to/output/dir \
    --model_name_or_path /path/to/previous_best_checkpoint \
    --model_type mt5 --gpus 8 \
    --do_train \
    --do_predict \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --max_source_length 1000  \
    --max_target_length 20 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --num_train_epochs 50 \
    --warmup_steps 500 
    --learning_rate 3e-05 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
``` 

- Train `mt5-large` based model. We train our mGEN on 8 GPUs with 24GB memory, and we found that we cannot train the model even with `train_batch_size==1` when we use adam optimizer. To fine-tune mt5-large based model, you have to set `--adafactor` option. 

```sh
python finetune_mgen.py \
    --data_dir /path/to/your/data/dir \
    --output_dir /path/to/model/output/dir \
    --model_name_or_path /path/to/previous_best_checkpoint \
    --model_type mt5 --gpus 8 \
    --do_train \
    --do_predict \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --max_source_length 800  \
    --max_target_length 20 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --num_train_epochs 50 \
    --warmup_steps 500 
    --learning_rate 3e-05 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --adafactor
``` 

### Evaluation

1. Run DPR
TO evaluate your trained mGEN model, you first need to retrieve passages using mDPR. Please follow the instruction in [mDPR](../mDPR) directory.

2. Convert DPR output
Please concert DPR output file as mentioned above.

3. Run mGEN
Please run the mGEN evaluation by running [`eval_mgen.py`](eval_mgen.py).

```
CUDA_VISIBLE_DEVICES=0 python eval_mgen.py \
    --model_name_or_path /path/to/model/output/dir \
    --evaluation_set /path/to/your/data/dir/val.source \
    --gold_data_path /path/to/your/data/dir/gold_para_qa_data_dev.tsv \
    --predictions_path mgen_output.txt \
    --gold_data_mode qa \
    --model_type mt5 \
    --max_length 20 \
    --eval_batch_size 8
```


