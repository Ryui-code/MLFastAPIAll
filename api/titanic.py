from fastapi import APIRouter
import joblib
from schemas.sch import TitanicSchema

titanic_router = APIRouter(prefix='/titanic-predict')

model = joblib.load('models/model (9).pkl')
scaler = joblib.load('models/scaler (9).pkl')

embarked_list = ['Q', 'S']

def build_features(t: TitanicSchema):
    numeric = [
        t.Age,
        t.SibSp,
        t.Parch,
        t.Pclass
    ]

    sex = [1 if t.Sex == 'male' else 0]

    embarked = [1 if t.Embarked == i else 0 for i in embarked_list]

    return numeric + sex + embarked

@titanic_router.post('/')
async def predict(titanic: TitanicSchema):
    features = build_features(titanic)

    scaled_data = scaler.transform([features])
    pred = int(model.predict(scaled_data)[0])

    return {'survived?': round(pred, 2)}