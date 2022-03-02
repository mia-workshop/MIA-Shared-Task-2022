# download models
mkdir models
wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_shared_task_all_langs_w100.tsv
wget https://nlp.cs.washington.edu/xorqa/cora/models/mGEN_model.zip
wget https://nlp.cs.washington.edu/xorqa/cora/models/mDPR_biencoder_best.cpt
unzip mGEN_model.zip
mkdir embeddings
cd embeddings
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_shared_task_embeddings/wiki_emb_en_$i 
done
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_shared_task_embeddings/wiki_emb_xor_$i  
done
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_shared_task_embeddings/wiki_others_emb__$i  
done
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/mia2022_shared_task_embeddings/wiki_others_emb_ms_tr_km_$i  
done
cd ../..

# Run mDPR
pip install transformers==3.0.2
cd mDPR
python dense_retriever.py \
    --model_file ../models/mDPR_biencoder_best.cpt \
    --ctx_file ../models/mia2022_shared_task_all_langs_w100.tsv \
    --qa_file ../data/eval/mia_2022_dev_xorqa.jsonl \
    --encoded_ctx_file "../models/embeddings/wiki_*" \
    --out_file xor_dev_dpr_retrieval_results.json \
    --n-docs 20 --validation_workers 1 --batch_size 256
cd ..

# Convert data 
cd mGEN
python3 convert_dpr_retrieval_results_to_seq2seq.py \
    --dev_fp ../mDPR/xor_dev_dpr_retrieval_results.json \
    --output_dir xorqa_dev_final_retriever_results \
    --top_n 15 \

# Run mGEN
pip install transformers==4.2.1
CUDA_VISIBLE_DEVICES=0 python eval_mgen.py \
    --model_name_or_path \
    --evaluation_set xorqa_dev_final_retriever_results/val.source \
    --gold_data_path xorqa_dev_final_retriever_results/gold_para_qa_data_dev.tsv \
    --predictions_path xor_dev_final_results.txt \
    --gold_data_mode qa \
    --model_type mt5 \
    --max_length 20 \
    --eval_batch_size 4
cd ..

# Run evaluation
cd eval_scripts
python eval_xor_full.py --data_file ../data/eval/mia_2022_dev_xorqa.jsonl --pred_file ../mGEN/xor_dev_final_results.txt --txt_file