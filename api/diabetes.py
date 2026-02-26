from fastapi import APIRouter
import joblib
from schemas.sch import DiabetesSchema
import uvicorn

diabetes_router = APIRouter(prefix='/diabetes-predict')

model = joblib.load('models/model (3).pkl')
scaler = joblib.load('models/scaler (3).pkl')

@diabetes_router.post('/')
async def predict(schema: DiabetesSchema):
    schema_dict = schema.dict()

    features = list(schema_dict.values())

    scaled_data = scaler.transform([features])
    diabetes = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    return {
        "diabetes": bool(diabetes),
        "probability": float(probability)
    }