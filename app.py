from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import os
from typing import List, Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

pd.set_option('future.no_silent_downcasting', True)

money_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

numeric_features = ['Num', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_features),
    ('categorical', categorical_pipeline, categorical_features)
])

def transformdata(data, trmode):

    cond = (data['CryoSleep'] == True)
    data.loc[cond, money_columns] = data.loc[cond, money_columns].fillna(0)
    data.loc[cond, ['VIP']] = data.loc[cond, ['VIP']].fillna(False)

    data[['Deck', 'Num', 'Side']] = data['Cabin'].str.split('/', expand=True)
    data.drop(columns=['Cabin'], inplace=True)

    cond = (data[money_columns].eq(0).all(axis=1))
    data.loc[cond, ['CryoSleep']] = data.loc[cond, ['CryoSleep']].fillna(True)

    data['Age'] = data['Age'].fillna(data.groupby(['HomePlanet', 'CryoSleep', 'VIP'])['Age'].transform('mean'))

    transformed_data = preprocessor.transform(data)

    if trmode == 1:
        traindata = data.drop("Transported", axis = 1)
        restraindata = data["Transported"]
        return [traindata, restraindata]
    return transformed_data

class Passenger(BaseModel):
    PassengerId: str
    HomePlanet: Optional[str]   
    CryoSleep: Optional[bool]
    Cabin: Optional[str]
    Destination: Optional[str]
    Age: Optional[float]
    VIP: Optional[bool]
    RoomService: Optional[float]
    FoodCourt: Optional[float]
    ShoppingMall: Optional[float]
    Spa: Optional[float]
    VRDeck: Optional[float]
    Name: Optional[str]

@app.post("/get_model")
async def get_model():
    if os.path.exists(MODEL_PATH):
        return Response(
            content=open(MODEL_PATH, 'rb').read(),
            media_type='application/octet-stream',
            headers={'Content-Disposition': f'attachment; filename="model.joblib"'}
        )
    else:
        return {"error": "Model file not found."}
    
traindata = pd.read_csv('train.csv')
tf_traindata = transformdata(traindata, trmode = 1)

@app.post("/predict")
async def predict(passengers: List[Passenger]):
    if model is None:
        return {"error": "Model not loaded."}
    
    passengers_data = [passenger.model_dump() for passenger in passengers]
    passengers_df = pd.DataFrame(passengers_data)
    passengers_ids = passengers_df['PassengerId']
    transformed_data = transformdata(passengers_df, trmode = 0)

    
    missing_columns = set(tf_traindata.columns) - set(passengers_df.columns)
    for column in missing_columns:
        transformed_data[column] = 0
    transformed_data = transformed_data[tf_traindata.columns]
    
    predictions = model.predict(transformed_data)
    
    output = pd.DataFrame({
        "PassengerId": passengers_ids,
        "Transported": str(bool(predictions))
    })
    return JSONResponse(output.to_dict(orient='records'))