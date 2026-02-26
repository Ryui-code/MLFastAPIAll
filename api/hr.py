import joblib
from fastapi import APIRouter
from schemas.sch import HrSchema

hr_router = APIRouter(prefix='/hr-predict')

model = joblib.load('models/svc_model.pkl')
scaler = joblib.load('models/svc_scaler.pkl')

job_role_list = ['Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director',
                 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative']

def build_features(hr: HrSchema):

    numeric = [
        hr.OverTime,
        hr.MonthlyIncome,
        hr.DistanceFromHome,
        hr.JobSatisfaction,
        hr.Age,
        hr.EnvironmentSatisfaction,
        hr.YearsAtCompany,
        hr.WorkLifeBalance
    ]

    overtime = [1 if hr.OverTime == 'Yes' else 0]

    job_role = [1 if hr.JobRole == i else 0 for i in job_role_list]

    return numeric + overtime + job_role

@hr_router.post('/')
async def predict(hr: HrSchema):
    features = build_features(hr)
    scaled_data = scaler.transform([features])
    pred = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    return {'attrition': bool(pred), 'probability': round(prob, 2)}