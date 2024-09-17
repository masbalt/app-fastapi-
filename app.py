from fastapi import FastAPI, Response
from pydantic import BaseModel
import joblib
import os
from typing import List, Optional
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

money_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
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

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

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

@app.post("/predict")
async def predict(passengers: List[Passenger]):
    if model is None:
        return {"error": "Model not loaded."}

    input_data_df = pd.DataFrame([passenger.dict() for passenger in passengers])
    
    cond = (input_data_df['CryoSleep'] == True)
    input_data_df.loc[cond, money_columns] = input_data_df.loc[cond, money_columns].fillna(0)
    input_data_df.loc[cond, ['VIP']] = input_data_df.loc[cond, ['VIP']].fillna(False)

    input_data_df[['Deck', 'Num', 'Side']] = input_data_df['Cabin'].str.split('/', expand=True)
    input_data_df.drop(columns=['Cabin'], inplace=True)

    cond = (input_data_df[money_columns].eq(0).all(axis=1))
    input_data_df.loc[cond, ['CryoSleep']] = input_data_df.loc[cond, ['CryoSleep']].fillna(True)

    input_data_df['Age'] = input_data_df['Age'].fillna(input_data_df.groupby(['HomePlanet', 'CryoSleep', 'VIP'])['Age'].transform('mean'))

    transformed_data = preprocessor.transform(input_data_df)

    predictions = model.predict(transformed_data)
    
    results = []
    for passenger, prediction in zip(passengers, predictions):
        results.append({
            "PassengerId": passenger.PassengerId,
            "Transported": str(bool(prediction))
        })

    return results