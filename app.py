from xgboost import XGBClassifier
import streamlit as st
import pickle as pk

model=pk.load(open("Churn_Predict.pkl","rb"))
st.set_page_config(page_title="Telecom Customer Churn Prediction",layout="centered")
st.title("Churn Prediction")
st.markdown("Predict whether a customer is likely to *churn or stay*")
st.divider()

st.subheader("Customer Details")

#input
gender=st.selectbox("Gender",["Male","Female"])
tenure=st.slider("No of months so far",0,72,12)
monthly_charges=st.number_input("Monthly Charges",0.0)
total_charges=st.number_input("Total Charges",0.0)
contract=st.selectbox("Contract type",["Month-to-month","One year","Two years"])
payment_method=st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer","Credit card"])
internet_service=st.selectbox("Internet Service",["None","DSL","Fiber optic"])
monthly_avg=total_charges/(tenure+1)
charges_ratio=monthly_charges/(total_charges+1)
tenure_charge_ratio=tenure/(monthly_charges+1)

def tenuregroup(tenure):
  if tenure<=6:
    return 0;
  elif 6<tenure<=24:
    return 1;
  else:
    return 2;

def autopay(payment_method):
  if payment_method in ["Bank transfer","Credit card"]:
    return 1;
  else:
    return 0;


def shortcontract(contract):
  if contract == "Month-to-month":
    return 1;
  else:
    return 0;

def fiberoptic(internet_service):
  if internet_service == "Fiber optic":
    return 1;
  else:
    return 0;

#features to pass to model
import numpy as np
gender_map={"Male":0,"Female":1}
tenure_group=tenuregroup(tenure)
is_autopay=autopay(payment_method)
short_contract=shortcontract(contract)
is_fiber=fiberoptic(internet_service)
contract_map={"Month-to-month":0,"One year":1,"Two years":2}
payment_method_map={"Electronic check":0,"Mailed check":1,"Bank transfer":2,"Credit card":3}
internet_service_map={"None":0,"DSL":1,"Fiber optic":2}
features=np.array([[gender_map[gender],1,1,1,tenure,1,1,internet_service_map[internet_service],1,1,1,0,1,0,contract_map[contract],0,payment_method_map[payment_method],
                  monthly_charges,total_charges,tenure_group,monthly_avg,is_autopay,6,short_contract,is_fiber,1,1,charges_ratio,
                  tenure_charge_ratio]],dtype=float)

#prediction
if st.button("Predict"):
  response=model.predict(features)
  if response==1:
    st.error("The customer is likely to **churn**")
  else:
    st.success("The customer is likely to **stay**")