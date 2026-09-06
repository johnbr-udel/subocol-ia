import boto3

from src.config import AWS_REGION


MODEL_ID = "amazon.nova-lite-v1:0"


def classify_claim_with_rag(
    current_claim_text,
    retrieved_claims
):
    client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION
    )

    historical_examples = ""

    for i, result in enumerate(retrieved_claims, start=1):
        historical_examples += f"""
--- Historical Claim {i} ---
{result["content"]["text"]}

"""

    prompt = f"""
    You are an insurance claim analyst.
    
    Your task is to determine whether the inspected vehicle damage is consistent
    with the reported accident.
    
    Possible classifications:
    
    OBJETADO:
    The inspected damage is not sufficiently consistent with the reported accident,
    there are suspicious inconsistencies, or there is insufficient evidence to
    confidently conclude that all inspected damage was caused by the reported event.
    
    ENTREGADO:
    The inspected damage is clearly and sufficiently consistent with the reported
    accident.
    
    IMPORTANT BUSINESS RULE:
    Failing to identify an OBJETADO claim is more costly than incorrectly reviewing
    an ENTREGADO claim.
    
    Therefore, if there is meaningful uncertainty about whether the inspected
    damage is explained by the reported accident, classify the claim as OBJETADO.
    
    Use the historical claims as reference examples, but make the final decision
    based primarily on the relationship between:
    
    1. The reported accident.
    2. The location and type of inspected damage.
    3. Whether the inspected parts are reasonably explained by the accident.
    
    CURRENT CLAIM:
    {current_claim_text}
    
    SIMILAR HISTORICAL CLAIMS:
    {historical_examples}
    
    Return only one word:
    
    OBJETADO
    
    or
    
    ENTREGADO
    """.strip() 

    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        inferenceConfig={
            "maxTokens": 20,
            "temperature": 0
        }
    )

    prediction = (
        response["output"]["message"]["content"][0]["text"]
        .strip()
        .upper()
    )

    return prediction