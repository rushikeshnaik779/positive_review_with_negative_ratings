import joblib 
import numpy as np 




def pred_naivebayes(df):
    print(df.head())
    pipeline = joblib.load('pipeline.pkl')
    df.drop('Developer Reply', aixs=1, inplace=True)
    df.dropna(inplace=True)
    print(df.head())
    df['pred'] = pipeline.predict(df['Text'])
    df['predprobab'] = pipeline.predict_proba(df['Text'])[:, 1]
    thr = 0.7
    df = df[(df['Star']==1) & (df['predprobab']>thr)] # thrshold 0.7 
    print('it worked till here')
    print(df)
    return df, thr 

