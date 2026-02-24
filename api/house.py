import joblib
from fastapi import APIRouter
from schemas.sch import HousePredictSchema

scaler = joblib.load('models/scaler (1).pkl')
model = joblib.load('models/model (1).pkl')

house_router = APIRouter(prefix='/house-predict', tags=['Predict Price'])

neighborhood_list = [
    'Blueste',
    'BrDale',
    'BrkSide',
    'ClearCr',
    'CollgCr',
    'Crawfor',
    'Edwards',
    'Gilbert',
    'IDOTRR',
    'MeadowV',
    'Mitchel',
    'NAmes',
    'NPkVill',
    'NWAmes',
    'NoRidge',
    'NridgHt',
    'OldTown',
    'SWISU',
    'Sawyer',
    'SawyerW',
    'Somerst',
    'StoneBr',
    'Timber',
    'Veenker',
]

@house_router.post('/')
async def predict_price(house: HousePredictSchema):
    house_dict = house.dict()
    new_neighborhood = house_dict.pop('Neighborhood')
    neighborhood1or_0 = [1 if new_neighborhood == i else 0 for i in neighborhood_list]

    features = list(house_dict.values()) + neighborhood1or_0
    scaled_data = scaler.transform([features])
    predict = model.predict(scaled_data)[0]
    return {'Price predict': predict}