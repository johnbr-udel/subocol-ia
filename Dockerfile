FROM public.ecr.aws/lambda/python:3.12

COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/

RUN pip install \
    --no-cache-dir \
    -r requirements-lambda.txt \
    --target ${LAMBDA_TASK_ROOT}

COPY src ${LAMBDA_TASK_ROOT}/src

CMD ["src.api.lambda_handler.lambda_handler"]