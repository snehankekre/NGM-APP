import streamlit as st
from PIL import Image

########################
# Page Title
########################

image=Image.open('https://storage.googleapis.com/afs-prod/media/media:94b8a45144234c3e8d495959903df184/2000.jpeg')

st.image(image, use_column_width=True)

st.write("""
# Breccia Rock Classification

This application helps to classify breccias into their different classes

***
""")
