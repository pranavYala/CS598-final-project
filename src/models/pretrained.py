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


DATA_ROOT = os.path.join(os.path.dirname(__file__), 'mimic3')

# how many rows to read for CHARTEVENTS chunks at a time
CHUNK_SIZE = 500_000

SAMPLE_NROWS = int(os.getenv('PREPROCESS_NROWS', 0)) or None


def load_csv(table_name):
    # load a CSV file from the MIMIC-III dataset
    # and return a DataFrame with lowercase column names
    lower = table_name.lower()
    for ext, comp in [('.csv', None), ('.csv.gz', 'gzip')]:
        path = os.path.join(DATA_ROOT, f'{table_name}{ext}')
        if not os.path.isfile(path):
            continue

        print(f"Loading {table_name} from {path}...")

        # for non-CHARTEVENTS tables, read the whole thing at once
        if lower not in ('chartevents',):
            df = pd.read_csv(
                path,
                compression=comp,
                nrows=SAMPLE_NROWS
            )
            # normalize to lowercase
            df.columns = df.columns.str.lower()

            if lower == 'admissions':
                df['admittime'] = pd.to_datetime(df['admittime'])
                df['dischtime'] = pd.to_datetime(df['dischtime'])

            return df

        #
        usecols = ['HADM_ID','CHARTTIME','ITEMID','VALUENUM']
        reader = pd.read_csv(
            path,
            compression=comp,
            usecols=usecols,
            parse_dates=['CHARTTIME'],
            engine='python',
            on_bad_lines='skip',
            chunksize=CHUNK_SIZE
        )

        chunks = []
        for chunk in reader:
            # lowercase column names immediately
            chunk.columns = [c.lower() for c in chunk.columns]
            # now chunk has columns 'hadm_id','charttime','itemid','valuenum'
            chunks.append(chunk)

            # if SAMPLE_NROWS is set, stop once we've read enough
            if SAMPLE_NROWS and sum(len(c) for c in chunks) >= SAMPLE_NROWS:
                break

        df = pd.concat(chunks, ignore_index=True)
        return df

    raise FileNotFoundError(f"Could not find {table_name}.csv(.gz) in {DATA_ROOT}")


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
    # extract time series data for a specific ITEMID
    df_v = df_chart[df_chart['itemid'] == itemid][['hadm_id','charttime','valuenum']].dropna()
    ts_dict = {}
    for hid, adm_time in df_adm[['hadm_id','admittime']].values:
        sub = df_v[df_v['hadm_id'] == hid]
        if sub.empty:
            ts_dict[hid] = [np.nan]*hours
            continue
        sub = sub.copy()
        sub['hr_since_admit'] = (sub['charttime'] - adm_time).dt.total_seconds()/3600
        sub = sub[sub['hr_since_admit'].between(0, hours)]
        sub['hr_hour'] = sub['hr_since_admit'].astype(int)
        hourly = sub.groupby('hr_hour')['valuenum'].mean()
        ts_dict[hid] = [float(hourly.get(i, np.nan)) for i in range(hours)]
    return ts_dict


def fill_ts(ts, sentinel=-1):
    ts = pd.Series(ts)
    if ts.notna().any():
        return ts.fillna(method='ffill').fillna(method='bfill').tolist()
    else:
        return [sentinel]*len(ts)


weak_label_fns = {
    'mortality': lambda r: int(r.get('hospital_expire_flag', 0)),
    'long_los':  lambda r: int((r['dischtime'] - r['admittime']).days >= 3),
}


def build_examples(df, text_cols, weak_labels=None):
    examples = []
    for _, row in df.iterrows():
        hid   = row['hadm_id']
        texts = [str(row.get(c, '')) for c in text_cols]
        text  = ' '.join([t for t in texts if t])
        labels = {}
        if weak_labels:
            for name, fn in weak_labels.items():
                labels[name] = fn(row)
        examples.append(Example(hid, text, labels))
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
    # 1) Load
    df_adm   = load_csv('ADMISSIONS')
    df_pat   = load_csv('PATIENTS')
    df_chart = load_csv('CHARTEVENTS')
    df_lab   = load_csv('LABEVENTS')
    df_dx    = load_csv('DIAGNOSES_ICD')

    # 2) Static aggregates
    dx_agg    = aggregate_diagnoses(df_dx)
    chart_agg = aggregate_chart(df_chart)
    lab_agg   = aggregate_labs(df_lab)

    print('Merging static tables...')
    df = df_adm.merge(dx_agg,    on='hadm_id', how='left')
    df = df.merge(chart_agg,     on='hadm_id', how='left')
    df = df.merge(lab_agg,       on='hadm_id', how='left')

    # fill NAs
    df['chart_mean_val'].fillna(df['chart_mean_val'].median(), inplace=True)
    df['lab_mean_val'].fillna(df['lab_mean_val'].median(),     inplace=True)

    # 3) Time-series
    ITEMID_HR  = 211
    ITEMID_SBP = 51
    ITEMID_DBP = 8368

    hr_dict  = extract_timeseries(df_chart, df_adm, ITEMID_HR,  hours=24)
    sbp_dict = extract_timeseries(df_chart, df_adm, ITEMID_SBP, hours=24)
    dbp_dict = extract_timeseries(df_chart, df_adm, ITEMID_DBP, hours=24)

    # 4) Text + weak labels
    examples = build_examples(df, ['diagnoses_text'], weak_labels=weak_label_fns)
    ids, in_ids, masks = tokenize_examples(examples, max_length=128)

    # assemble final DataFrame
    out = pd.DataFrame({
        'hadm_id':        ids,
        'input_ids':      in_ids,
        'attention_mask': masks,
    })
    for name in weak_label_fns:
        out[name] = [ex.labels[name] for ex in examples]

    df_idx = df.set_index('hadm_id')
    out['chart_mean_val'] = out['hadm_id'].map(df_idx['chart_mean_val'])
    out['lab_mean_val']   = out['hadm_id'].map(df_idx['lab_mean_val'])
    out['heart_rate_ts']  = out['hadm_id'].map(lambda h: fill_ts(hr_dict[h]))
    out['sbp_ts']         = out['hadm_id'].map(lambda h: fill_ts(sbp_dict[h]))
    out['dbp_ts']         = out['hadm_id'].map(lambda h: fill_ts(dbp_dict[h]))

    os.makedirs('preprocessed', exist_ok=True)
    out_path = os.path.join('preprocessed', 'mimic_examples.parquet')
    out.to_parquet(out_path, index=False)
    print(f"Saved {len(out)} examples to {out_path}")

    with open('mimic_examples_head.txt','w') as f:
        f.write(out.head().to_string(index=False))


if __name__ == '__main__':
    main()
