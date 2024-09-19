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
import logging

pd.set_option('future.no_silent_downcasting', True)

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




logging.basicConfig(
    level=logging.INFO,  # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),  # Запись логов в файл
        logging.StreamHandler()  # Вывод логов в консоль
    ]
)
logger = logging.getLogger(__name__)

money_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
numeric_features = ['Num', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']

app = FastAPI()

#загружаем модель и препроцессор
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'voting_classifier.pkl')

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

preprocessor = joblib.load(os.path.join(os.path.dirname(__file__), 'preprocessor.pkl'))

def transformdata(data):
    def split_cabin(cabin):
        if pd.isna(cabin):
            return pd.Series({'Deck': None, 'Num': None, 'Side': None})
        try:
            deck, num, side = cabin.split('/')
            return pd.Series({'Deck': deck, 'Num': num, 'Side': side})
        except:
            return pd.Series({'Deck': None, 'Num': None, 'Side': None})

    logger.info('Transforming data for prediction.')

    cond = (data['CryoSleep'] == True)
    data.loc[cond, money_columns] = data.loc[cond, money_columns].fillna(0)
    data.loc[cond, ['VIP']] = data.loc[cond, ['VIP']].fillna(False)

    cond = (data[money_columns].eq(0).all(axis=1))
    data.loc[cond, ['CryoSleep']] = data.loc[cond, ['CryoSleep']].fillna(True)
    data['CryoSleep'].fillna(False, inplace=True)

    age_means = data.groupby(['HomePlanet', 'CryoSleep', 'VIP'])['Age'].transform('mean')
    data['Age'] = data['Age'].fillna(age_means)
    

    cabin_split = data['Cabin'].apply(split_cabin)
    data = pd.concat([data, cabin_split], axis=1)
    data.drop(columns=['Cabin'], inplace=True)
    data['Num'] = pd.to_numeric(data['Num'], errors='coerce')

    data.drop(columns=['Name', 'PassengerId'], inplace=True)

    logger.info('Data transformed successfully.')
    return data


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
    logger.info('Received prediction request.')
    if model is None or preprocessor is None:
        logger.error('Model or preprocessor not loaded.')
        return {"error": "Model or preprocessor not loaded."}

    try:
        passengers_data = [passenger.model_dump() for passenger in passengers]
        passengers_df = pd.DataFrame(passengers_data)
        logger.info('Dataframe created from passengers data.')

        passengers_ids = passengers_df['PassengerId']

        transformed_data = transformdata(passengers_df)
        
        transformed_data = preprocessor.transform(transformed_data)
        logger.info('Data preprocessed successfully.')

        predictions = model.predict(transformed_data)
        logger.info('Predictions made successfully.')

        output = pd.DataFrame({
            "PassengerId": passengers_ids,
            "Transported": predictions
        })
        output['Transported'] = output['Transported'].apply(lambda x: 'True' if x else 'False')

        return JSONResponse(content=output.to_dict(orient='records'))

    except Exception as e:
        logger.error(f'Error occurred during prediction: {e}')
        return {"error": "An error occurred during prediction."}