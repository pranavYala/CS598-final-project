## CS598-final-project

This repo reproduces **“A Comprehensive EHR Timeseries Pre-training Benchmark”** : https://dl.acm.org/doi/pdf/10.1145/3450439.3451877

# Setup

**Install dependencies**  
pip install -r requirements.txt


**Download Instructions for MIMIC-III (v1.4)**
Create a PhysioNet account
Complete CITI Training and sign a data agreement on PhysioNet
Download the data using this command from the terminal (you will be prompted for your PhysioNet password):
wget -r -N -c -np --user <user-name> --ask-password https://physionet.org/files/mimiciii/1.4/ 


1. **Preprocess and save MIMIC examples**
python data/preprocess.py

2. **Turn the Parquet into BERT+TS embeddings**
python bert_model/load_and_yield_embeddings.py

3. **Run run_model**
python run_model.py \
  --embeddings preprocessed/mimic_embeddings_with_timeseries.npy \
  --parquet    preprocessed/mimic_examples.parquet \
  --output     preprocessed/weak_pretrained_model.pt \
  --batch_size 64 \
  --epochs     10 \
  --lr         1e-5

4. **Run evaluator**
python experiments/evaluator.py \
  --embeddings preprocessed/mimic_embeddings_with_timeseries.npy \
  --parquet    preprocessed/mimic_examples.parquet \
  --model_path preprocessed/weak_pretrained_model.pt \
  --batch_size 64
