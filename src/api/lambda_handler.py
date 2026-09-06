import json

from src.api.schemas import ClaimRequest

from src.api.inference import (
    prepare_claim_for_inference,
    predict_with_baseline,
    predict_with_rag,
    download_model_from_s3
)

from src.config import (
    S3_BUCKET,
    BEDROCK_KB_ID
)


BASELINE_MODEL_KEY = "models/baseline_bundle.joblib"

RAG_TOP_K = 5


def lambda_handler(event, context):

    # Read the JSON body sent by API Gateway
    body = event.get("body", {})

    if isinstance(body, str):
        body = json.loads(body)

    # Validate the incoming claim
    claim = ClaimRequest(**body)

    # Convert the request into the format expected by our models
    claim_df = prepare_claim_for_inference(
        claim
    )

    # Identify which endpoint was called
    path = event.get("rawPath", "")

    # --------------------------------------------------
    # Baseline prediction
    # --------------------------------------------------
    if path == "/predict/baseline":

        model_path = download_model_from_s3(
            bucket=S3_BUCKET,
            key=BASELINE_MODEL_KEY
        )

        baseline_result = predict_with_baseline(
            claim_df=claim_df,
            model_path=model_path
        )

        response = {
            "model": "baseline",
            "prediction": baseline_result["prediction"],
            "probability_objetado": baseline_result[
                "probability_objetado"
            ]
        }

    # --------------------------------------------------
    # RAG prediction
    # --------------------------------------------------
    elif path == "/predict/rag":

        rag_result = predict_with_rag(
            claim_df=claim_df,
            knowledge_base_id=BEDROCK_KB_ID,
            number_of_results=RAG_TOP_K
        )

        response = {
            "model": "rag",
            "prediction": rag_result["prediction"],
            "retrieved_claims": rag_result[
                "retrieved_claims"
            ]
        }

    # --------------------------------------------------
    # Run both models
    # --------------------------------------------------
    elif path == "/predict/compare":

        model_path = download_model_from_s3(
            bucket=S3_BUCKET,
            key=BASELINE_MODEL_KEY
        )

        baseline_result = predict_with_baseline(
            claim_df=claim_df,
            model_path=model_path
        )

        rag_result = predict_with_rag(
            claim_df=claim_df,
            knowledge_base_id=BEDROCK_KB_ID,
            number_of_results=RAG_TOP_K
        )

        response = {
            "baseline": {
                "prediction": baseline_result["prediction"],
                "probability_objetado": baseline_result[
                    "probability_objetado"
                ]
            },
            "rag": {
                "prediction": rag_result["prediction"],
                "retrieved_claims": rag_result[
                    "retrieved_claims"
                ]
            }
        }

    # --------------------------------------------------
    # Unknown endpoint
    # --------------------------------------------------
    else:
        return {
            "statusCode": 404,
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps({
                "error": "Endpoint not found"
            })
        }

    # IMPORTANT:
    # This return belongs to ALL successful branches above.
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(response)
    }