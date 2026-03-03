# app.py

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components 
def main():
    st.title("Student CGPA Predictor")
    st.write("This is a placeholder for the Streamlit app. The actual implementation will be in `app.py`.")
    
if __name__ == "__main__":
    main()

data = pd.read_csv("data/Student_data.csv")

st.dataframe(data.head())

with open("rapport_analyse_exploratoire.html", "r", encoding="utf-8") as html_file:
    rapport_html = html_file.read()

components.html(rapport_html, height=600, scrolling=True)

# test