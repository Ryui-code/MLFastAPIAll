import joblib
from fastapi import APIRouter
from schemas.sch import HrSchema

hr_router = APIRouter(prefix='/hr-predict')

model = joblib.load('models/model (5).pkl')
scaler = joblib.load('models/scaler (5).pkl')

job_role_list = ['Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director',
                 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative']
business_travel_list = ['Travel_Frequently', 'Travel_Rarely']
department_list = ['Research & Development', 'Sales']
education_list = ['Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree']
marital_list = ['Married', 'Single']

DEFAULTS = {
    "Education": 3,
    "HourlyRate": 50,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "MonthlyIncome": 5000,
    "MonthlyRate": 20000,
    "NumCompaniesWorked": 2,
    "PercentSalaryHike": 12,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 2,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 2,
    "BusinessTravel": "Travel_Rarely",
    "Department": "Research & Development",
    "EducationField": "Life Sciences",
    "Gender": "Male",
    "MaritalStatus": "Single"
}

def build_features(hr: HrSchema):

    numeric = [
        hr.Age,
        hr.DailyRate,
        hr.DistanceFromHome,
        hr.Education,
        hr.EnvironmentSatisfaction,
        hr.HourlyRate,
        hr.JobInvolvement,
        hr.JobLevel,
        hr.JobSatisfaction,
        hr.MonthlyIncome,
        hr.MonthlyRate,
        hr.NumCompaniesWorked,
        hr.PercentSalaryHike,
        hr.PerformanceRating,
        hr.RelationshipSatisfaction,
        hr.StockOptionLevel,
        hr.TotalWorkingYears,
        hr.TrainingTimesLastYear,
        hr.WorkLifeBalance,
        hr.YearsAtCompany,
        hr.YearsInCurrentRole,
        hr.YearsSinceLastPromotion,
        hr.YearsWithCurrManager
    ]

    overtime = [1 if hr.OverTime == 'Yes' else 0]
    job_role = [1 if hr.JobRole == role else 0 for role in job_role_list]
    business_travel = [1 if hr.BusinessTravel == bt else 0 for bt in business_travel_list]
    department = [1 if hr.Department == d else 0 for d in department_list]
    education_field = [1 if hr.EducationField == ef else 0 for ef in education_list]
    gender = [1 if hr.Gender == 'male' else 0]
    marital_status = [1 if hr.MaritalStatus == ms else 0 for ms in marital_list]

    return (
        numeric
        + overtime
        + job_role
        + business_travel
        + department
        + education_field
        + gender
        + marital_status
    )

@hr_router.post('/')
async def predict(hr: HrSchema):
    features = build_features(hr)
    scaled_data = scaler.transform([features])
    pred = model.predict(scaled_data)[0]
    prob = model.predict_proba(scaled_data)[0][1]

    return {'attrition': bool(pred), 'probability': round(prob, 2)}