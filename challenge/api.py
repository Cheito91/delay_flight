import fastapi
import pandas as pd
from fastapi import HTTPException, status, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import List
from challenge.model import DelayModel

app = fastapi.FastAPI()


# Custom exception handler to return 400 instead of 422 for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )

# Initialize and train the model once at startup
model = DelayModel()

# Define valid values based on the dataset
VALID_OPERAS = [
    "Aerolineas Argentinas", "Aeromexico", "Air Canada", "Air France", "Alitalia",
    "American Airlines", "Austral", "Avianca", "British Airways", "Copa Air",
    "Delta Air", "Gol Trans", "Grupo LATAM", "Iberia", "JetSmart SPA",
    "K.L.M.", "Lacsa", "Latin American Wings", "Oceanair Linhas Aereas",
    "Plus Ultra Lineas Aereas", "Qantas Airways", "Sky Airline", "United Airlines"
]

VALID_TIPOVUELO = ["N", "I"]
VALID_MES = list(range(1, 13))  # 1 to 12


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator('MES')
    def validate_mes(cls, v):
        if v not in VALID_MES:
            raise ValueError(f'MES must be between 1 and 12, got {v}')
        return v

    @validator('TIPOVUELO')
    def validate_tipovuelo(cls, v):
        if v not in VALID_TIPOVUELO:
            raise ValueError(f'TIPOVUELO must be "N" or "I", got {v}')
        return v

    @validator('OPERA')
    def validate_opera(cls, v):
        if v not in VALID_OPERAS:
            raise ValueError(f'Invalid OPERA: {v}')
        return v


class PredictRequest(BaseModel):
    flights: List[Flight]


class PredictResponse(BaseModel):
    predict: List[int]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> PredictResponse:
    try:
        # Convert flights to DataFrame
        flights_data = [flight.dict() for flight in request.flights]
        df = pd.DataFrame(flights_data)
        
        # Preprocess the data
        features = model.preprocess(df)
        
        # Make predictions
        predictions = model.predict(features)
        
        return PredictResponse(predict=predictions)
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )