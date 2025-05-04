import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import sys

# ------------- CONFIGURATION -------------
LAB_VITAL_BUCKETS = {
    # example: map raw item IDs to bucket names
    # 50820: 'Glucose', 50931: 'Glucose', …
}
TREATMENT_VARS = {
    'ventilation': ['ventilator', 'respirator'],  # example keywords
    'vasopressor': ['norepinephrine', 'epinephrine'],
    'fluid_bolus': ['bolus']
}
STATIC_COLS = ['age', 'gender', 'ethnicity', 'insurance', 'admission_type', 'first_care_unit']
OUTPUT_DIR = Path('data/processed')
# -----------------------------------------

def bucket_labs_vitals(df, bucket_map):
    """Map raw lab/vital item IDs into named buckets."""
    df['bucket'] = df['itemid'].map(bucket_map)
    return df.dropna(subset=['bucket'])

def pivot_and_hourly(df, time_col='charttime', value_col='valuenum'):
    """
    Pivot to wide form by bucket, then resample to hourly bins.
    """
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    wide = df.pivot_table(index='subject_id', columns='bucket', values=value_col, aggfunc='mean')
    # Resample per ICU stay will be handled outside
    return wide

def extract_cohort_mimic(raw_dir, output_dir=OUTPUT_DIR):
    """
    1) Load ICU stays and filter: first ICU stay of adult patients.
    2) Load labevents, chartevents; bucket & pivot.
    3) Load inputevents_*; identify treatments.
    4) Load patients/admissions for static features.
    5) For each stay: merge timeseries, resample hourly, impute, save.
    """
    os.makedirs(output_dir, exist_ok=True)
    # 1) ICU stays & patients
    icustays = pd.read_csv(raw_dir / 'ICUSTAYS.csv')
    patients = pd.read_csv(raw_dir / 'PATIENTS.csv')
    admissions = pd.read_csv(raw_dir / 'ADMISSIONS.csv')
    # filter adults and first ICU stay
    stays = icustays.merge(patients[['subject_id','dob']], on='subject_id')
    stays['age'] = (pd.to_datetime(stays['intime']) - pd.to_datetime(stays['dob'])).dt.days/365.25
    stays = stays[stays['age']>=15].sort_values(['subject_id','intime']).drop_duplicates('subject_id')
    # 2) Labs & vitals
    labevents = pd.read_csv(raw_dir / 'LABEVENTS.csv')
    chartevents = pd.read_csv(raw_dir / 'CHARTEVENTS.csv')
    lv = pd.concat([labevents, chartevents], ignore_index=True)
    lv = bucket_labs_vitals(lv, LAB_VITAL_BUCKETS)
    # 3) Treatments
    input_cv = pd.read_csv(raw_dir / 'INPUTEVENTS_CV.csv')
    input_iv = pd.read_csv(raw_dir / 'INPUTEVENTS_MV.csv')
    inputs = pd.concat([input_cv, input_iv], ignore_index=True)
    # tag treatments by keywords
    for trt, kws in TREATMENT_VARS.items():
        inputs[trt] = inputs['itemid'].astype(str).str.contains('|'.join(kws), case=False)
    # 4) Static
    stat = admissions[['subject_id','insurance','admission_type']]\
        .merge(patients[['subject_id','gender','ethnicity']], on='subject_id')
    # 5) For each stay, build timeseries
    for _, stay in stays.iterrows():
        sid, intime, outtime = stay.subject_id, stay.intime, stay.outtime
        mask = (lv.subject_id==sid) & (lv.charttime.between(intime, outtime))
        df_lv = lv[mask].copy()
        df_lv['charttime'] = pd.to_datetime(df_lv['charttime'])
        df_lv = df_lv.set_index('charttime').groupby('bucket').resample('H')['valuenum'].mean().unstack(0)
        # treatments
        df_tr = inputs[inputs.subject_id==sid].copy()
        df_tr['starttime'] = pd.to_datetime(df_tr['starttime'])
        df_tr = df_tr.set_index('starttime')[list(TREATMENT_VARS)].resample('H').max().fillna(0)
        # merge static features
        row_static = stat[stat.subject_id==sid].iloc[0].to_dict()
        df_static = pd.DataFrame([row_static]*len(df_lv), index=df_lv.index)
        # full
        df_full = pd.concat([df_lv, df_tr, df_static], axis=1).sort_index()
        # impute missing: forward/backward fill then mean
        df_full = df_full.ffill().bfill().fillna(df_full.mean())
        # time-since-last-measured
        ts = df_full.index.to_series().diff().fillna(pd.Timedelta(seconds=0)).dt.seconds / 3600
        df_full['time_since_last'] = ts.cumsum()
        # save
        out_path = output_dir / f'mimic_cohort_{sid}.npz'
        np.savez_compressed(out_path,
                            data=df_full.iloc[:].values,
                            columns=df_full.columns.values)
    print("MIMIC preprocessing done.")

def extract_cohort_eicu(raw_dir, output_dir=OUTPUT_DIR):
    """
    Analogous pipeline for eICU (different table names, filtering).
    """
    # similar implementation for eICU-CRD
    pass

if __name__=='__main__':
    raw = Path(sys.argv[1])
    extract_cohort_mimic(raw)
    # extract_cohort_eicu(raw)  # once you have eICU files

