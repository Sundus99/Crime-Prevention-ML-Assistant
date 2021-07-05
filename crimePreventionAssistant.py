#-------------------------------------------------------------------------------
# Name:        Crime Prevention ML Assistant
# Purpose:     Detect, report and deter crimes
#
# Author:      Sundus Yawar
#
# Created:     Jul-04-2021
# Copyright:   (c) Sundus Yawar 2021
#-------------------------------------------------------------------------------
import streamlit as st
from PIL import Image
from model import *
import pandas as pd
import altair as alt
import streamlit.components.v1 as components
#==============libs for sending email============
import smtplib
import os
from dotenv import load_dotenv
from datetime import datetime


load_dotenv()
email = os.getenv('EMAIL')
password = os.getenv('PASSWORD')
#================================================

st.sidebar.markdown("Welcome to Crime Prevention Assisstant.")

st.title("Model trained with Google's Teachable Machine")

st.write("")
#st.write("Armed: This part is trained with images of different race, gender, positons and locations in which a person is holding a gun.")
#st.write("")
page = st.selectbox('Page',('Crime Prevention Assistant','Why is it needed?','About Data Training Set'))
if page == "Why is it needed?":
    image = Image.open('PoliceTorontoShootingData.png')
    st.image(image, caption='City of Toronto Shootings Stats - These shootings exclude suicide and firearm discharged by police',use_column_width=True)
    st.write("Source: https://data.torontopolice.on.ca/pages/shootings")
    st.write("We are just in the middle of 2021 and there are already 174 shooting incidents. Which is alarming. Hopefully, this bot can help find and prevent these incidents from occuring too frequently.")

if page == "About Data Training Set":
    st.header("About Data Training Set")
    a = "Armed: This part is trained with images of different race, gender, positons and locations in which a person is holding a gun."
    b = "Unarmed: This part is trained in a similar manner. However, it also includes poses that are similar to holding a gun but in which the person is giving directions. And includes people walking, being on phone, running, sitting."
    st.markdown(f'<p style="background-color:#c60404;color:#ffffff;font-size:16px;border-radius:2%;">{a}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#c60404;color:#ffffff;font-size:16px;border-radius:2%;">{b}</p>', unsafe_allow_html=True)
    st.write("")
    c= "With police's cooperation to improve the dataset by giving CCTV footage in which gun violence occured, the model can vastly improve. As it will be exposed to the environment it will be mnitoring and hence be able to detect someone taking out a gun or pointing a gun in an actual setting efficiently. As currently, it is trained to detect gun in ideal condition of sorts. (not on street, close ups etc)"
    st.markdown(f'<p style="background-color:#c60404;color:#ffffff;font-size:16px;border-radius:2%;">{c}</p>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f'<p style="background-color:#c60404;color:#ffffff;font-size:16px;border-radius:2%;">{"On field, it will classify with live camera."}</p>', unsafe_allow_html=True)
    
if page == 'Crime Prevention Assistant':
    st.header("Classifies armed and unarmed person")
    st.write("Upload images of arm or unarmed person for quick analysis.")
    # file upload and handling logic
    uploaded_file = st.file_uploader("Choose a image with armed or unarmed person", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Image Uploaded for Crime Detection.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machines_model(image, 'keras_model.h5')
        #when I downloaded the model I trained on teachable machine, it gave a file
        #called labels.txt that mentions which number represents which class.
        if label == 0:#0 is armed
            st.write("The image depicts person holding a gun.")
            with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
                smtp.ehlo()#identifies with mail server
                smtp.starttls()#establish encrypt traffic
                smtp.ehlo()#reidentify as encrypted connection
                
                #now can login to mail server
                smtp.login(email, password)

                now = datetime.now()
                date = now.strftime("%b-%d-%Y")
                time = now.strftime("%H:%M:%S")

                subject = 'Emergency@JaneSt and Finch Ave W - CrimePreventionAssistant'#format emergency@(name of intersection the camera is near)
                body = f"Someone armed spotted in the area today ({date}) @ {time}" + "\n Courtesy of Crime Prevention ML Assistant"
                msg = f'Subject: {subject} \n\n{body}'

                #now send email
                smtp.sendmail(email,email,msg)#sender,receiver,msg
            audio_file = open('policesiren.mp3','rb')
            audio_bytes = audio_file.read()
            st.write(" 🚨🚓 Authorities have been notified via crime watch email! 🚨🚓 ")
            st.audio(audio_bytes,format('audio/mp3'))
        elif label == 1:#1 is unarmed
            st.write(" 🕊️ The image depicts no criminal activity. 😌 ")