
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Visit with Us | Wellness Package', page_icon='✈️')
st.title('Wellness Tourism Package propensity')
st.caption('Prioritize outreach using a predicted probability of purchase.')

MODEL_PATH = Path(__file__).with_name('wellness_propensity_model.joblib')
if not MODEL_PATH.exists():
    st.error('Model not found. Run the GitHub Actions pipeline before deploying the app.')
    st.stop()
model = joblib.load(MODEL_PATH)

with st.form('customer_form'):
    age = st.number_input('Age', 18, 100, 35)
    contact = st.selectbox('Type of contact', ['Self Enquiry', 'Company Invited'])
    city_tier = st.selectbox('City tier', [1, 2, 3])
    duration = st.number_input('Pitch duration (minutes)', 0.0, value=15.0)
    occupation = st.selectbox('Occupation', ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    visiting = st.number_input('People visiting', 1, 10, 2)
    followups = st.number_input('Follow-ups', 0, 20, 3)
    product = st.selectbox('Product pitched', ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
    stars = st.selectbox('Preferred property star', [3, 4, 5])
    marital = st.selectbox('Marital status', ['Single', 'Married', 'Divorced', 'Unmarried'])
    trips = st.number_input('Trips per year', 0.0, value=3.0)
    passport = st.selectbox('Has passport', [0, 1])
    satisfaction = st.selectbox('Pitch satisfaction score', [1, 2, 3, 4, 5])
    car = st.selectbox('Owns a car', [0, 1])
    children = st.number_input('Children visiting', 0.0, value=0.0)
    designation = st.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
    income = st.number_input('Monthly income', 0.0, value=20000.0)
    submitted = st.form_submit_button('Score customer')

if submitted:
    customer = pd.DataFrame([{
        'Age': age, 'TypeofContact': contact, 'CityTier': city_tier,
        'DurationOfPitch': duration, 'Occupation': occupation, 'Gender': gender,
        'NumberOfPersonVisiting': visiting, 'NumberOfFollowups': followups,
        'ProductPitched': product, 'PreferredPropertyStar': stars,
        'MaritalStatus': marital, 'NumberOfTrips': trips, 'Passport': passport,
        'PitchSatisfactionScore': satisfaction, 'OwnCar': car,
        'NumberOfChildrenVisiting': children, 'Designation': designation,
        'MonthlyIncome': income,
    }])
    probability = float(model.predict_proba(customer)[0, 1])
    st.metric('Purchase likelihood', f'{probability:.1%}')
    st.success('Prioritize for outreach' if probability >= 0.50 else 'Add to a lower-cost nurture journey')
