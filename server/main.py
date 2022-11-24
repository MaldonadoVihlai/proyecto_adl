import streamlit as st
import os
from PIL import Image
from helper import *
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
sns.set()
import zipfile
import constants
import shutil

get_top_page_content(st)

uploaded_zip_file = st.file_uploader("Cargue la carpeta con las imágenes", type='zip')
if uploaded_zip_file is not None:
    zf = zipfile.ZipFile(uploaded_zip_file)
    zf.extractall(constants.EXTRACTION_DIRECTORY)
    file_name = uploaded_zip_file.name.split('.')[0]    
    y_pred = process_dataset()
    print(y_pred)
    st.markdown('## Predicción :')
    st.markdown(get_mgmt_state(y_pred[0][0]))
    shutil.rmtree(constants.EXTRACTION_DIRECTORY+'/')
    #fig, ax = plt.subplots()
    #ax  = sns.barplot(y = 'name',x='values', data = y_pred,order = y_pred.sort_values('values',ascending=False).name)
    #ax.set(xlabel='Confidence %', ylabel='Breed')

    #st.pyplot(fig)
    


