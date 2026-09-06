import boto3
import sys 
import pandas as pd 

PROJECT_ROOT="/home/sagemaker-user/subocol-ia"


sys.path.append(PROJECT_ROOT)

from src.config import AWS_REGION 

def get_s3_client():
    return boto3.client("s3",region_name=AWS_REGION)


def get_object_metadata(bucket, key):
    s3 = get_s3_client()
    response = s3.head_object(
        Bucket=bucket,
        Key=key
    )
    return response


def load_csv_from_s3(bucket, key):
    s3 = get_s3_client()

    response = s3.get_object(
        Bucket=bucket,
        Key=key
    )

    df = pd.read_csv(response["Body"],sep=";")

    return df

def upload_text_to_s3(bucket, key, text):
    s3 = get_s3_client()

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType="text/plain"
    )

def upload_documents_to_s3(bucket, prefix, documents):
    for claim_id, document in documents.items():
        key = f"{prefix}/{claim_id}.txt"

        upload_text_to_s3(
            bucket,
            key,
            document
        )

def list_s3_objects(bucket, prefix):
    s3 = get_s3_client()

    paginator = s3.get_paginator("list_objects_v2")

    objects = []

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            objects.append(obj["Key"])

    return objects

def upload_file_to_s3(local_path, bucket, key):
    s3 = get_s3_client()

    s3.upload_file(
        str(local_path),
        bucket,
        key
    )