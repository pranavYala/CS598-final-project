import os
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import BertTokenizer

class Example:
    def __init__(self, admission_id, text, labels=None):
        self.admission_id = admission_id
        self.text = text
        self.labels = labels or {}

def load_csv(table_name):
    path = os.path.join('mimic-iii-clinical-database-demo-1.4',
                        'mimic-iii-clinical-database-demo-1.4',
                        f'{table_name}.csv')
    print(f'Loading {table_name}...')
    df = pd.read_csv(path)
    if table_name == 'ADMISSIONS':
        df['admittime'] = pd.to_datetime(df['admittime'])
        df['dischtime'] = pd.to_datetime(df['dischtime'])
    return df

def aggregate_diagnoses(df_dx):
    print('Aggregating diagnoses...')
    agg = (
        df_dx
        .groupby('hadm_id')['icd9_code']
        .agg(lambda codes: ' '.join(map(str, codes)))
        .reset_index()
        .rename(columns={'icd9_code': 'diagnoses_text'})
    )
    return agg

def aggregate_chart(df_chart):
    print('Aggregating chart events...')
    numeric = df_chart[['hadm_id', 'valuenum']].dropna()
    agg = (
        numeric
        .groupby('hadm_id')['valuenum']
        .mean()
        .reset_index()
        .rename(columns={'valuenum': 'chart_mean_val'})
    )
    return agg

def aggregate_labs(df_lab):
    print('Aggregating lab events...')
    numeric = df_lab[['hadm_id', 'valuenum']].dropna()
    agg = (
        numeric
        .groupby('hadm_id')['valuenum']
        .mean()
        .reset_index()
        .rename(columns={'valuenum': 'lab_mean_val'})
    )
    return agg

def extract_timeseries(df_chart, df_adm, itemid, hours=24):
    df_vital = df_chart[df_chart['itemid'] == itemid]
    df_vital = df_vital[['hadm_id', 'charttime', 'valuenum']].dropna()
    df_vital['charttime'] = pd.to_datetime(df_vital['charttime'])

    ts_dict = {}
    for hadm_id, adm_time in df_adm[['hadm_id', 'admittime']].values:
        rows = df_vital[df_vital['hadm_id'] == hadm_id]
        if rows.empty:
            ts_dict[hadm_id] = [np.nan] * hours
            continue
        rows = rows.copy()
        rows['hr_since_admit'] = (rows['charttime'] - adm_time).dt.total_seconds() / 3600
        rows = rows[rows['hr_since_admit'].between(0, hours)]
        rows['hr_hour'] = rows['hr_since_admit'].astype(int)
        hourly = rows.groupby('hr_hour')['valuenum'].mean()
        ts = [float(hourly.get(i, np.nan)) for i in range(hours)]
        ts_dict[hadm_id] = ts
    return ts_dict

def fill_ts(ts, sentinel=-1):
    ts = pd.Series(ts)
    if ts.notna().any():
        return ts.fillna(method='ffill').fillna(method='bfill').tolist()
    else:
        return [sentinel] * len(ts)

weak_label_fns = {
    'mortality': lambda r: int(r.get('hospital_expire_flag', 0)),
    'long_los': lambda r: int((r['dischtime'] - r['admittime']).days >= 3)
}

def build_examples(df, text_cols, weak_labels=None):
    examples = []
    for _, row in df.iterrows():
        adm_id = row['hadm_id']
        texts = [str(row.get(col, '')) for col in text_cols]
        text = ' '.join([t for t in texts if t])
        labels = {}
        if weak_labels:
            for name, fn in weak_labels.items():
                labels[name] = fn(row)
        examples.append(Example(adm_id, text, labels))
    return examples

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_examples(examples, max_length=128):
    ids, input_ids_list, masks_list = [], [], []
    for ex in examples:
        enc = tokenizer.encode_plus(
            ex.text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True
        )
        ids.append(ex.admission_id)
        input_ids_list.append(enc['input_ids'])
        masks_list.append(enc['attention_mask'])
    return ids, input_ids_list, masks_list

def main():
    df_adm   = load_csv('ADMISSIONS')
    df_dx    = load_csv('DIAGNOSES_ICD')
    df_chart = load_csv('CHARTEVENTS')
    df_lab   = load_csv('LABEVENTS')

    dx_agg    = aggregate_diagnoses(df_dx)
    chart_agg = aggregate_chart(df_chart)
    lab_agg   = aggregate_labs(df_lab)

    print('Merging tables...')
    df = df_adm.merge(dx_agg,    on='hadm_id', how='left')
    df = df.merge(chart_agg, on='hadm_id', how='left')
    df = df.merge(lab_agg,   on='hadm_id', how='left')

    df['chart_mean_val'].fillna(df['chart_mean_val'].median(), inplace=True)
    df['lab_mean_val'].fillna(df['lab_mean_val'].median(),     inplace=True)

    ITEMID_HR = 211
    ITEMID_SBP = 51
    ITEMID_DBP = 8368

    hr_ts_dict  = extract_timeseries(df_chart, df_adm, itemid=ITEMID_HR,  hours=24)
    sbp_ts_dict = extract_timeseries(df_chart, df_adm, itemid=ITEMID_SBP, hours=24)
    dbp_ts_dict = extract_timeseries(df_chart, df_adm, itemid=ITEMID_DBP, hours=24)

    text_columns = ['diagnoses_text']
    examples = build_examples(df, text_columns, weak_labels=weak_label_fns)
    ids, input_ids, masks = tokenize_examples(examples, max_length=128)

    out = pd.DataFrame({
        'hadm_id': ids,
        'input_ids': input_ids,
        'attention_mask': masks,
    })
    for name in weak_label_fns:
        out[name] = [ex.labels[name] for ex in examples]

    df_indexed = df.set_index('hadm_id')
    out['chart_mean_val'] = out['hadm_id'].map(df_indexed['chart_mean_val'])
    out['lab_mean_val']   = out['hadm_id'].map(df_indexed['lab_mean_val'])

    out['heart_rate_ts']  = out['hadm_id'].map(lambda hid: fill_ts(hr_ts_dict[hid]))
    out['sbp_ts']         = out['hadm_id'].map(lambda hid: fill_ts(sbp_ts_dict[hid]))
    out['dbp_ts']         = out['hadm_id'].map(lambda hid: fill_ts(dbp_ts_dict[hid]))

    os.makedirs('preprocessed', exist_ok=True)
    out_path = os.path.join('preprocessed', 'mimic_examples.parquet')
    out.to_parquet(out_path, index=False)

    print(f"Saved {len(out)} examples to {out_path}")
    with open('mimic_examples_head.txt', 'w') as f:
        f.write(out.head().to_string(index=False))

if __name__ == '__main__':
    main()
