import streamlit as st
from PIL import Image

########################
# Page Title
########################

image=Image.Open("gold.2-1.jpg")

st.image(image, use_column_width=True)

st.write("""
# Breccia Rock Classification

This application helps to classify breccias into their different classes

***
""")
