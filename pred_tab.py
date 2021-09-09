import joblib 
import numpy as np 




def pred_naivebayes(df):
    print(df.shape)
    pipeline = joblib.load('pipeline.pkl')
    df.dropna(inplace=True)
    print(df.shape)
    df['pred'] = pipeline.predict(df['Text'])
    df['predprobab'] = pipeline.predict_proba(df['Text'])[:, 1]
    thr = 0.85
    df = df[(df['Star']==1) & (df['predprobab']>thr)] # thrshold 0.7 
    print('it worked till here')
    print(df)
    return df, thr 

