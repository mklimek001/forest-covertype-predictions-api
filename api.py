import uvicorn
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from keras.models import load_model
from simple_models import heuristic

app = FastAPI()

scaler = joblib.load('./models/scaler.save')

model_rf = joblib.load('./models/model_rf.save')
model_kn = joblib.load('./models/model_kn.save')
model_nn = load_model('./models/model_nn.save')


class CompleteSample(BaseModel):
    Elevation: int
    Aspect: int
    Slope: int
    Horizontal_Distance_To_Hydrology: int
    Vertical_Distance_To_Hydrology: int
    Horizontal_Distance_To_Roadways: int
    Hillshade_9am: int
    Hillshade_Noon: int
    Hillshade_3pm: int
    Horizontal_Distance_To_Fire_Points: int
    Wilderness_Area_0: int = 0
    Wilderness_Area_1: int = 0
    Wilderness_Area_2: int = 0
    Wilderness_Area_3: int = 0
    Soil_Type_0: int = 0
    Soil_Type_1: int = 0
    Soil_Type_2: int = 0
    Soil_Type_3: int = 0
    Soil_Type_4: int = 0
    Soil_Type_5: int = 0
    Soil_Type_6: int = 0
    Soil_Type_7: int = 0
    Soil_Type_8: int = 0
    Soil_Type_9: int = 0
    Soil_Type_10: int = 0
    Soil_Type_11: int = 0
    Soil_Type_12: int = 0
    Soil_Type_13: int = 0
    Soil_Type_14: int = 0
    Soil_Type_15: int = 0
    Soil_Type_16: int = 0
    Soil_Type_17: int = 0
    Soil_Type_18: int = 0
    Soil_Type_19: int = 0
    Soil_Type_20: int = 0
    Soil_Type_21: int = 0
    Soil_Type_22: int = 0
    Soil_Type_23: int = 0
    Soil_Type_24: int = 0
    Soil_Type_25: int = 0
    Soil_Type_26: int = 0
    Soil_Type_27: int = 0
    Soil_Type_28: int = 0
    Soil_Type_29: int = 0
    Soil_Type_30: int = 0
    Soil_Type_31: int = 0
    Soil_Type_32: int = 0
    Soil_Type_33: int = 0
    Soil_Type_34: int = 0
    Soil_Type_35: int = 0
    Soil_Type_36: int = 0
    Soil_Type_37: int = 0
    Soil_Type_38: int = 0
    Soil_Type_39: int = 0


class SimplifiedSample(BaseModel):
    Elevation: int
    Aspect: int
    Slope: int
    Horizontal_Distance_To_Hydrology: int
    Vertical_Distance_To_Hydrology: int
    Horizontal_Distance_To_Roadways: int
    Hillshade_9am: int
    Hillshade_Noon: int
    Hillshade_3pm: int
    Horizontal_Distance_To_Fire_Points: int
    Wilderness_Area: int
    Soil_Type: int


class_dict = {1: 'Spruce/Fir',
              2: 'Lodgepole Pine',
              3: 'Ponderosa Pine',
              4: 'Cottonwood/Willow',
              5: 'Aspen',
              6: 'Douglas-fir',
              7: 'Krummholz'}


def predict_class(sample, model, data):
    sample = pd.DataFrame(scaler.transform(sample), columns=sample.columns)
    if model == 'heuristic':
        class_num = heuristic(data)
    elif model == 'neighbors':
        class_num = model_kn.predict(sample)[0].item()
    elif model == 'random_forest':
        class_num = model_kn.predict(sample)[0].item()
    elif model == 'neural':
        nn_predictions = model_nn.predict(sample)
        class_num = nn_predictions.argmax(axis=-1)[0].item()
    else:
        raise HTTPException(status_code=422, detail="Model name must be in "
                                                    "[heuristic, neighbors, random_forest, neural]")

    return {'class_number': class_num,
            'forest_type': class_dict[class_num]}


@app.get('/')
def index():
    return {'message': 'This is API for predicting forest cover type from cartographic data.'}


@app.post('/predict/{model}')
def predict_review(model: str, data: CompleteSample):
    data = data.dict()
    sample = pd.DataFrame(data, index=[0])
    return predict_class(sample, model, data)


@app.post('/predict-simple-sample/{model}')
def predict_review(model: str, simple_data: SimplifiedSample):
    simple_data = simple_data.dict()

    data = CompleteSample(
        Elevation=simple_data['Elevation'],
        Aspect=simple_data['Aspect'],
        Slope=simple_data['Slope'],
        Horizontal_Distance_To_Hydrology=simple_data['Horizontal_Distance_To_Hydrology'],
        Vertical_Distance_To_Hydrology=simple_data['Vertical_Distance_To_Hydrology'],
        Horizontal_Distance_To_Roadways=simple_data['Horizontal_Distance_To_Roadways'],
        Hillshade_9am=simple_data['Hillshade_9am'],
        Hillshade_Noon=simple_data['Hillshade_Noon'],
        Hillshade_3pm=simple_data['Hillshade_3pm'],
        Horizontal_Distance_To_Fire_Points=simple_data['Horizontal_Distance_To_Fire_Points']
    )

    data = data.dict()
    sample = pd.DataFrame(data, index=[0])

    if simple_data['Wilderness_Area'] in [i for i in range(4)]:
        sample[f"Wilderness_Area_{str(simple_data['Wilderness_Area'])}"] = 1

    if simple_data['Soil_Type'] in [i for i in range(40)]:
        sample[f"Soil_Type_{str(simple_data['Soil_Type'])}"] = 1

    return predict_class(sample, model, data)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
