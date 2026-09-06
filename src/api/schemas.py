from datetime import datetime

from pydantic import BaseModel, Field


class ClaimRequest(BaseModel):
    version_hechos: str

    piezas: list[str] = Field(
        min_length=1
    )

    marca: str
    linea: str
    version: str
    modelo: int
    fecha_creacion: datetime


class PredictionResponse(BaseModel):
    model: str
    prediction: str


class BaselinePredictionResponse(PredictionResponse):
    probability_objetado: float