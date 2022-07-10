import streamlit as st
from PIL import Image

########################
# Page Title
########################

image=Image.open(r"C:\Users\tosun\PycharmProjects\pythonProject1\gold.2-1.jpg")

st.image(image, use_column_width=True)

st.write("""
# Breccia Rock Classification

This application helps to classify breccias into their different classes

***
""")
