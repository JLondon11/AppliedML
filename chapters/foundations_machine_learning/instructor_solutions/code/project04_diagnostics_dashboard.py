"""Streamlit dashboard for tidy optimization metrics."""
import argparse

def main():
    import pandas as pd, streamlit as st
    ap=argparse.ArgumentParser(add_help=False); ap.add_argument('--csv',default='metrics.csv'); args,_=ap.parse_known_args()
    st.title('Optimization diagnostics'); df=pd.read_csv(args.csv); st.dataframe(df.tail())
    x='step' if 'step' in df else df.columns[0]
    for c in ['loss','val_metric','grad_norm','memory_bytes','communication_bytes','utilization']:
        if c in df: st.subheader(c); st.line_chart(df.set_index(x)[c])
if __name__=='__main__': main()
