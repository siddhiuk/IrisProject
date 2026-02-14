import pickle
import numpy as np
import pandas as pd
import streamlit as st

#load the scaler and model
def predict_species(sep_len,sep_width,pet_len,pet_width,scaler_path,model_path):
        try:
            with open(scaler_path,'rb') as file1:
                scaler=pickle.load(file1)
            with open(model_path,'rb') as file2:
                model=pickle.load(file2)

            dct={
                'SepalLengthCm':[sep_len],
                'SepalWidthCm':[sep_width],
                'PetalLengthCm':[pet_len],
                'PetalWidthCm':[pet_width]
            }    

            x_new=pd.DataFrame(dct)
            # Ensure column order matches what the scaler expects
            x_new = x_new[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

            xnew_pre=scaler.transform(x_new)

            #make predictions
            pred=model.predict(xnew_pre)
            prob=model.predict_proba(xnew_pre)
            max_prob=np.max(prob)

            return pred,max_prob
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None, None
        
import streamlit as st


st.title("Iris Species Prediction")

sep_len=st.number_input("Enter sepal length",min_value=0.0,step=0.1,value=5.1)
sep_width=st.number_input("Enter sepal width",min_value=0.0,step=0.1,value=3.5)
pet_len=st.number_input("Enter petal length",min_value=0.0,step=0.1,value=1.4)
pet_width=st.number_input("Enter petal width",min_value=0.0,step=0.1,value=0.2) 

if st.button("Predict"):
    scaler_path='notebook/scaler.pkl'
    model_path='notebook/model.pkl'

    pred,max_prob=predict_species(sep_len,sep_width,pet_len,pet_width,scaler_path,model_path)
    if pred is not None and max_prob is not None:
        st.success(f"Predicted Species: {pred[0]}, Probability: {max_prob:.2f}")
    else:
        st.error("Prediction failed. Please check the input values and try again.")