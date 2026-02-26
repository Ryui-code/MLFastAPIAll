from fastapi import APIRouter
import joblib
from schemas.sch import TelecomSchema

telecom_router = APIRouter(prefix='/telecom-predict')

model = joblib.load('models/model (7).pkl')
scaler = joblib.load('models/scaler (7).pkl')

MultipleLines_list = ['No phone service', 'Yes']
InternetService_list = ['Fiber optic', 'No']
OnlineSecurity_list = ['No internet service', 'Yes']
OnlineBackup_list = ['No internet service', 'Yes']
DeviceProtection_list = ['No internet service', 'Yes']
TechSupport_list = ['No internet service', 'Yes']
StreamingTV_list = ['No internet service', 'Yes']
StreamingMovies_list = ['No internet service', 'Yes']
Contract_list = ['One year', 'Two year']
PaymentMethod_list = ['Credit card (automatic)', 'Electronic check', 'Mailed check']

def build_features(t: TelecomSchema):

    numeric = [
        t.SeniorCitizen,
        t.tenure,
        t.MonthlyCharges,
        t.TotalCharges
    ]

    gender = [1 if t.gender == 'male' else 0]

    partner = [1 if t.Partner == 'Yes' else 0]

    dependents = [1 if t.Dependents == 'Yes' else 0]

    phone_service = [1 if t.PhoneService == 'Yes' else 0]

    multiple_lines = [1 if t.MultipleLines == i else 0 for i in MultipleLines_list]

    internet_service = [1 if t.InternetService == i else 0 for i in InternetService_list]

    online_security = [1 if t.OnlineSecurity == i else 0 for i in OnlineSecurity_list]

    online_backup = [1 if t.OnlineBackup == i else 0 for i in OnlineBackup_list]

    device_protection = [1 if t.DeviceProtection == i else 0 for i in DeviceProtection_list]

    tech_support = [1 if t.TechSupport == i else 0 for i in TechSupport_list]

    streaming_tv = [1 if t.StreamingTV == i else 0 for i in StreamingTV_list]

    streaming_movies = [1 if t.StreamingMovies == i else 0 for i in StreamingMovies_list]

    contract = [1 if t.Contract == i else 0 for i in Contract_list]

    paper_less_billing = [1 if t.PaperlessBilling == 'Yes' else 0]

    payment_method = [1 if t.PaymentMethod == i else 0 for i in PaymentMethod_list]

    return (
        numeric
        + gender
        + partner
        + dependents
        + phone_service
        + multiple_lines
        + internet_service
        + online_security
        + online_backup
        + device_protection
        + tech_support
        + streaming_tv
        + streaming_movies
        + contract
        + paper_less_billing
        + payment_method
    )

@telecom_router.post('/')
async def predict(telecom: TelecomSchema):
    features = build_features(telecom)
    scaled_data = scaler.transform([features])

    pred = model.predict(scaled_data)[0]
    prob = float(model.predict_proba(scaled_data)[0][1])

    return {
        "churn": bool(pred),
        "probability": round(prob, 2)
    }