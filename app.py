import streamlit as st
import pandas as pd 
import base64

from pred_tab import pred_naivebayes


def get_table_download(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="output.csv">Download csv file</a>'
    return href


if __name__ == "__main__":
    df = None
    data_file = st.file_uploader("Upload CSV",type=['csv'])
    st.write(data_file)
    if data_file is not None:
        file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
        #st.write(file_details)
        df = pd.read_csv(data_file)
        if df is not None:
            pass 


    
    if st.button("Show Data"):
        if df is not None:
            st.dataframe(df[['Text', 'Star']])
            df, thr = pred_naivebayes(df[['Text', 'Star']])
            st.write(f"# Results:  with threshold {thr*100}%" )
            st.dataframe(df)
            st.markdown(get_table_download(df), unsafe_allow_html=True)



                





