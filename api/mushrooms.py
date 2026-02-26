from fastapi import APIRouter
import joblib
import uvicorn
from schemas.sch import MushroomSchema

mushrooms_router = APIRouter(prefix='/mushrooms-predict')

log_reg = joblib.load('models/model (6).pkl')
tree = joblib.load('models/tree_model.pkl')
scaler = joblib.load('models/scaler (6).pkl')

cap_shape_list = ["c", "f", "k", "s", "x"]
cap_surface_list = ["g", "s", "y"]
cap_color_list = ["c", "e", "g", "n", "p", "r", "u", "w", "y"]
odor_list = ["c", "f", "l", "m", "n", "p", "s", "y"]
gill_color_list = ["e", "g", "h", "k", "n", "o", "p", "r", "u", "w", "y"]
stalk_root_list = ["c", "e", "r"]
stalk_surface_list = ['k', 's', 'y']
stalk_color_list = ["c", "e", "g", "n", "o", "p", "w", "y"]
veil_color_list = ["o", "w", "y"]
ring_number_list = ['o', 't']
ring_type_list = ['f', 'l', 'n', 'p']
spore_print_color_list = ['h', 'k', 'n', 'o', 'r', 'u', 'w', 'y']
population_list = ['c', 'n', 's', 'v', 'y']
habitat_list = ['g', 'l', 'm', 'p', 'u', 'w']

def build_features(m: MushroomSchema):

    cap_shape = [
        1 if m.cap_shape == i else 0 for i in cap_shape_list
    ]

    cap_surface = [
        1 if m.cap_surface == i else 0 for i in cap_surface_list
    ]

    cap_color = [
        1 if m.cap_color == i else 0 for i in cap_color_list
    ]

    bruises = [1 if m.bruises == "t" else 0]

    odor = [
        1 if m.odor == i else 0 for i in odor_list
    ]

    gill_attachment = [1 if m.gill_attachment == "f" else 0]

    gill_spacing = [1 if m.gill_spacing == "w" else 0]

    gill_size = [1 if m.gill_size == "n" else 0]

    gill_color = [
        1 if m.gill_color == i else 0 for i in gill_color_list
    ]

    stalk_shape = [1 if m.stalk_shape == "t" else 0]

    stalk_root = [
        1 if m.stalk_root == i else 0 for i in stalk_root_list
    ]

    stalk_surface_above = [
        1 if m.stalk_surface_above_ring == i else 0 for i in stalk_surface_list
    ]

    stalk_surface_below = [
        1 if m.stalk_surface_below_ring == i else 0 for i in stalk_surface_list
    ]

    stalk_color_above = [
        1 if m.stalk_color_above_ring == i else 0 for i in stalk_color_list
    ]

    stalk_color_below = [
        1 if m.stalk_color_below_ring == i else 0 for i in stalk_color_list
    ]

    veil_color = [
        1 if m.veil_color == i else 0 for i in veil_color_list
    ]

    ring_number = [
        1 if m.ring_number == i else 0 for i in ring_number_list
    ]

    ring_type = [
        1 if m.ring_type == i else 0 for i in ring_type_list
    ]

    spore_print = [
        1 if m.spore_print_color == i else 0 for i in spore_print_color_list
    ]

    population = [
        1 if m.population == i else 0 for i in population_list
    ]

    habitat = [
        1 if m.habitat == i else 0 for i in habitat_list
    ]

    return (
        cap_shape
        + cap_surface
        + cap_color
        + bruises
        + odor
        + gill_attachment
        + gill_spacing
        + gill_size
        + gill_color
        + stalk_shape
        + stalk_root
        + stalk_surface_above
        + stalk_surface_below
        + stalk_color_above
        + stalk_color_below
        + veil_color
        + ring_number
        + ring_type
        + spore_print
        + population
        + habitat
    )

@mushrooms_router.post('/predict-logistic')
async def predict_logistic(m: MushroomSchema):
    features = build_features(m)
    scaled_data = scaler.transform([features])
    pred = log_reg.predict(scaled_data)[0]
    prob = log_reg.predict_proba(scaled_data)[0][1]

    return {'poisonous': bool(pred), 'probability': float(prob)}

@mushrooms_router.post('/predict-tree')
async def predict_tree(m: MushroomSchema):
    features = build_features(m)
    pred = tree.predict([features])[0]
    prob = tree.predict_proba([features])[0][1]

    return {'poisonous': bool(pred), 'probability': float(prob)}