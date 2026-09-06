import joblib
import numpy as np
import pandas as pd
import boto3

from pathlib import Path

from src.baseline.features import transform_features
from src.config import AWS_REGION

from src.rag.prepare_documents import build_claim_query
from src.rag.retrieve import retrieve_similar_claims
from src.rag.classify import classify_claim_with_rag

from src.baseline.features import transform_features


def prepare_claim_for_inference(claim):
    data = claim.model_dump()

    piezas = [
        pieza.strip()
        for pieza in data["piezas"]
        if pieza.strip()
    ]

    fecha_creacion = pd.to_datetime(
        data["fecha_creacion"]
    )

    vehicle_age = max(
        fecha_creacion.year - data["modelo"],
        0
    )

    parts_text = " ".join(piezas)

    claim_data = {
        "version_hechos": data["version_hechos"],
        "parts_text": parts_text,
        "marca": data["marca"],
        "linea": data["linea"],
        "version": data["version"],
        "modelo": data["modelo"],
        "fecha_creacion": fecha_creacion,
        "vehicle_age": vehicle_age,
        "piezas_totales": len(piezas),
        "valid_parts": len(piezas)
    }

    return pd.DataFrame([claim_data])

def predict_with_baseline(claim_df, model_path):
    bundle = joblib.load(model_path)

    model = bundle["model"]
    narrative_vectorizer = bundle["narrative_vectorizer"]
    parts_vectorizer = bundle["parts_vectorizer"]
    brand_encoder = bundle["brand_encoder"]
    numeric_scaler = bundle["numeric_scaler"]
    threshold = bundle["threshold"]

    X = transform_features(
        claim_df,
        narrative_vectorizer,
        parts_vectorizer,
        brand_encoder,
        numeric_scaler
    )

    objetado_index = list(
        model.classes_
    ).index("OBJETADO")

    probability_objetado = model.predict_proba(
        X
    )[0, objetado_index]

    prediction = (
        "OBJETADO"
        if probability_objetado >= threshold
        else "ENTREGADO"
    )

    return {
        "prediction": prediction,
        "probability_objetado": float(probability_objetado)
    }   

def predict_with_rag(
    claim_df,
    knowledge_base_id,
    number_of_results=5
):
    row = claim_df.iloc[0]

    query = build_claim_query(row)

    retrieved_claims = retrieve_similar_claims(
        query_text=query,
        knowledge_base_id=knowledge_base_id,
        number_of_results=number_of_results
    )

    prediction = classify_claim_with_rag(
        current_claim_text=query,
        retrieved_claims=retrieved_claims
    )

    return {
        "prediction": prediction,
        "retrieved_claims": len(retrieved_claims)
    }


def download_model_from_s3(
    bucket,
    key,
    local_path="/tmp/baseline_bundle.joblib"
):
    local_path = Path(local_path)

    if not local_path.exists():
        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION
        )

        s3.download_file(
            bucket,
            key,
            str(local_path)
        )

    return local_path