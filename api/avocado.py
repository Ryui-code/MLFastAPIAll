from fastapi import APIRouter
import joblib
from schemas.sch import AvocadoSchema

avocado_router = APIRouter(prefix='/avocado-predict')

model = joblib.load('models/model (8).pkl')
scaler = joblib.load('models/scaler (8).pkl')

@avocado_router.post('/')
async def predict(avocado: AvocadoSchema):
    avocado_dict = avocado.dict()

    new_color = avocado_dict.pop('color_category')
    cat1or_0 = [
        1 if new_color == 'dark green' else 0,
        1 if new_color == 'green' else 0,
        1 if new_color == 'purple' else 0
    ]

    features = list(avocado_dict.values()) + cat1or_0

    scaled_data = scaler.transform([features])

    predict_class = model.predict(scaled_data)[0]

    proba = model.predict_proba(scaled_data)[0]

    class_names = model.classes_

    probabilities = {i: round(prob, 2) for i, prob in zip(class_names, proba)}

    return {
        "predicted_ripeness": predict_class,
        "probabilities": probabilities
    }