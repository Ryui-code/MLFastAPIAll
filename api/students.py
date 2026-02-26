from fastapi import APIRouter
import joblib
import uvicorn
from schemas.sch import StudentSchema

students_router = APIRouter(prefix='/students-predict')

model = joblib.load('models/model (4).pkl')
scaler = joblib.load('models/scaler (4).pkl')

@students_router.post('/')
async def predict(student: StudentSchema):
    student_dict = student.dict()

    new_gender = student_dict.pop('gender')
    gender1or_0 = [
        1 if new_gender == 'male' else 0,
    ]

    new_race = student_dict.pop('race_ethnicity')
    race1or_0 = [
        1 if new_race == 'group B' else 0,
        1 if new_race == 'group C' else 0,
        1 if new_race == 'group D' else 0,
        1 if new_race == 'group E' else 0,
    ]

    new_parent = student_dict.pop('parental_level_of_education')
    parent1or_0 = [
        1 if new_parent == "bachelor's degree" else 0,
        1 if new_parent == 'high school' else 0,
        1 if new_parent == "master's degree" else 0,
        1 if new_parent == 'some college' else 0,
        1 if new_parent == 'some high school' else 0
    ]

    new_lunch = student_dict.pop('lunch')
    lunch1or_0 = [
        1 if new_lunch == 'standard' else 0,
    ]

    new_test_preparation_course = student_dict.pop('test_preparation_course')
    test1or_0 = [
        1 if new_test_preparation_course == 'none' else 0,
    ]

    features = list(student_dict.values()) + gender1or_0 + race1or_0 + parent1or_0 + lunch1or_0 + test1or_0

    scaled_data = scaler.transform([features])
    predict_ = model.predict(scaled_data)[0]
    return {'predict': round(predict_, 2)}