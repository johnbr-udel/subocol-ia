import boto3

from src.config import AWS_REGION


def retrieve_similar_claims(
    query_text,
    knowledge_base_id,
    number_of_results=5
):
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name=AWS_REGION
    )

    response = client.retrieve(
        knowledgeBaseId=knowledge_base_id,
        retrievalQuery={
            "text": query_text
        },
        retrievalConfiguration={
            "vectorSearchConfiguration": {
                "numberOfResults": number_of_results
            }
        }
    )

    return response["retrievalResults"]

import re


def get_historical_label(result):
    text = result["content"]["text"]

    match = re.search(
        r"Decisión histórica:\s*(OBJETADO|ENTREGADO)",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return None


def select_balanced_claims(
    retrieved_claims,
    per_class=3
):
    selected = {
        "OBJETADO": [],
        "ENTREGADO": []
    }

    for result in retrieved_claims:

        label = get_historical_label(result)

        if (
            label in selected
            and len(selected[label]) < per_class
        ):
            selected[label].append(result)

    balanced_results = (
        selected["OBJETADO"]
        + selected["ENTREGADO"]
    )

    balanced_results = sorted(
        balanced_results,
        key=lambda x: x["score"],
        reverse=True
    )

    return balanced_results