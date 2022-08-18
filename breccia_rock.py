import altair as alt
import streamlit as st
from PIL import Image
import base64
import io
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow 
import os
########################
# Page Title
########################

st.set_page_config(page_title='Breccia Classifier', page_icon = './Extras/ngm.jfif', layout = 'wide', initial_sidebar_state = 'expanded')

image = Image.open('./Extras/nevadaimage.jfif')
st.image(image, use_column_width=True)

st.write("""
# BRECCIA ROCK CLASSIFICATION

This application helps to classify breccias into their different classes

***
""")

# pylint: enable=line-too-long
@st.cache
#defining a function to load image
def load_image(image_file):
    img=Image.open(image_file)
    return img



#creating a directory
dir = "Breccia_Rock"
try:
    os.mkdir(dir)
    print ("Directory is created")
except FileExistsError:
    print ("Directory already exists")
    
    

#image Preprocessing
batch_size=1

def pre_process():
    Breccia_Predict_Images = os.getcwd()
    Breccia_Predict_datagen = ImageDataGenerator(rescale=1. / 255)
    img_width, img_height = 400, 600
    Breccia_predict_generator = Breccia_Predict_datagen.flow_from_directory(Breccia_Predict_Images, target_size=(img_width, img_height),
                                                                            batch_size=1, class_mode=None, shuffle=False)
    return Breccia_predict_generator


#grabbing names from working directory and removing extension
from natsort import natsorted
Image_dir=os.getcwd()
Breccia_name= []
for root, dirs, files in os.walk(Image_dir):
    j=natsorted(files)
    for i in j:
        if i.endswith('jpg'):
            Breccia_name.append(i)
            #print(i)
            #Breccia_name.extend(os.path.splitext(name)[0] for name in i)
print(Breccia_name)
      


#Grabbing the image name and depth

spliited, name, GeoFrom, GeoTo=[],[],[],[]
for i in Breccia_name:
# setting the maxsplit parameter to 3, will return a list with 4 elements!    
    x = i.split("_", 3)
    spliited.append(x)
#print(spliited)

for j in spliited:
    name_=j[0]
    GeoFrom_=j[1]
    GeoTo_=j[2]
    name.append(name_)
    GeoFrom.append(GeoFrom_)
    GeoTo.append(GeoTo_)


#loading the model
#model_path='.\\Extras\\Breccia_Rock_Classifier.h5'



#predictions
@st.cache
def Breccia_Predictions():
    image_=pre_process()
    model = tensorflow.keras.models.load_model('./Extras/Breccia_Rock_Classifier.h5','r')
    prediction_steps_per_epoch = np.math.ceil(image_.n / image_.batch_size)
    image_.reset()
    Breccia_predictions = model.predict_generator(image_, steps=prediction_steps_per_epoch, verbose=1)
    model.close()
    predicted_classes = np.argmax(Breccia_predictions, axis=1)
    return predicted_classes


#Application Main Body

def main():
    image_file=st.file_uploader('upload a breccia rock', type=['png', 'jpg', 'jpeg'], accept_multiple_files=True,)
    if image_file is not None:
        for image in image_file:
            file_details = {"FileName":image.name,"FileType":image.type}
            st.write(file_details)
            img = load_image(image)
            st.image(img)
            #saving file
            with open(os.path.join("Breccia_Rock",image.name),"wb") as f: 
                f.write(image.getbuffer())         
            st.success("Saved File")
            
                
        #predict Button        
                
        if(st.button('Predict')):
            predicted=Breccia_Predictions()
            list_predicted_classes=predicted.tolist()
            Final_prediction1=pd.DataFrame(data=zip(name, GeoFrom, GeoTo, list_predicted_classes),columns=['HoleID','GeoFrom', 'GeoTo','Predicted_Labels'])
            Final_prediction1['Predicted_Labels']= Final_prediction1['Predicted_Labels'].replace({0: '1BX', 1:'2BX', 2: '3BX', 
                                                                                    3: 'CAVBX', 4: 'FBX', 5: 'MBX', 6: 'NBX'}).astype(str)
            st.dataframe(Final_prediction1)
            
            st.success("Prediction Successful")
            for i in image_file:
                os.remove('Breccia_Rock/'+i.name)

                
        
if __name__ == "__main__":
    main()
    
    
    
