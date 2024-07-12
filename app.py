import streamlit as st
import pandas as pd
import pickle
Email_Type_option={'Marketing':1,'Important Updates':2}
Email_Source_Type_option={'Sales and Marketing':1,'Important Admin':2}
Time_Email_sent_Category_option={'Morning':1,'Evening':2,'Night':3}
Email_Campaign_Type_option={'Type 1':1,'Type 2':2,'Type 3':4}
Customer_Location_option={'Mumbai':'G', 'Hyderabad':'C', 'Bengaluru':'E','Kolkata': 'A','Pune' : 'B','Jaipur': 'F', 'New Delhi':'D'}
st.title('Email Campaign Effectiveness Prediction')
col1,col2,col3=st.columns(3)
with col1:
    Email_Type=st.selectbox('Email Type',['Marketing','Important Updates'])
with col2:
    Time_Email_sent_Category = st.selectbox('Email sent Time', ['Morning','Evening','Night'])
with col3:
    Email_Campaign_Type=st.selectbox('Email Campaign Type',['Type 1','Type 2','Type 3'])
col4,col5=st.columns(2)
with col4:
    Email_Source_Type = st.selectbox('Email Source Type', ['Sales and Marketing', 'Important Admin'])
with col5:
    Customer_Location=st.selectbox('Customer Location',sorted(['Mumbai','Hyderabad','Bengaluru','Kolkata','Pune','Jaipur','New Delhi']))
Subject_Hotness_Score=st.slider('Subject Hotness Score',min_value=0.0,max_value=5.0)
col6,col7=st.columns(2)
with col6:
    Total_Past_Communications=st.number_input('Total Past Communications',placeholder="Type a number...",value=None,min_value=0)
with col7:
    Word_Count=st.number_input('Total Words',placeholder="Type a number...",value=None,min_value=0)
col8,col9=st.columns(2)
with col8:
    Total_Links=st.number_input('Total Links',placeholder="Type a number...",value=None,min_value=0)
with col9:
    Total_Images=st.number_input('Total Images',placeholder="Type a number...",value=None,min_value=0)

param_for_XGB={
        'min_child_weight': [0,1, 5],
        'gamma': [0.5, 1],
        'subsample': [0.5,0.6, 0.8],
        'colsample_bytree': [0.8,0.9],
        'max_depth': [5,6]
        }

CT = pickle.load(open('CT.pickle','rb'))
model = pickle.load(open('model.pickle','rb'))

if st.button('Prediction'):
    df=pd.DataFrame({
        'Email_Type':[Email_Type_option[Email_Type]], 'Subject_Hotness_Score':[Subject_Hotness_Score], 'Email_Source_Type':[Email_Source_Type_option[Email_Source_Type]],
           'Customer_Location':[Customer_Location_option[Customer_Location]], 'Email_Campaign_Type':[Email_Campaign_Type_option[Email_Campaign_Type]], 'Total_Past_Communications':[Total_Past_Communications],
           'Time_Email_sent_Category':[Time_Email_sent_Category_option[Time_Email_sent_Category]], 'Word_Count':[Word_Count], 'Total_Links':[Total_Links],
           'Total_Images':[Total_Images]
    })
    df = CT.transform(df)
    probability = model.predict_proba(df)

    p0=f'{round((probability[0][0]),2)*100} %'
    p1=f'{round((probability[0][1]),2)*100} %'
    p2=f'{round((probability[0][2]),2)*100} %'

    result={
        0: ['Ignored',p0],
        1: ['Read',p1],
        2: ['Acknowledged',p2]
    }
    pred = model.predict(df)
    st.header(f'Email would be: {(result[pred[0]])[0]}')
    st.header(f'Probability of {(result[pred[0]])[0]} is {(result[pred[0]])[1]}')
