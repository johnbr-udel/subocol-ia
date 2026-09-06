def build_claim_document(row):
    document = f"""
ID del aviso: {row['numero_aviso']}

Fecha de creación: {row['fecha_creacion']}

Vehículo:
Marca: {row['marca']}
Línea: {row['linea']}
Versión: {row['version']}
Modelo: {row['modelo']}
Edad del vehículo: {row['vehicle_age']}

Versión de los hechos:
{row['version_hechos']}

Piezas inspeccionadas:
{row['parts_text']}

Número total de piezas: {row['piezas_totales']}
Número de piezas con descripción válida: {row['valid_parts']}

Decisión histórica:
{row['estado_aviso']}
""".strip()

    return document


def build_training_documents(train_df):
    documents = {}

    for _, row in train_df.iterrows():
        claim_id = row["numero_aviso"]

        documents[claim_id] = build_claim_document(row)

    return documents


def build_claim_query(row):
    query = f"""
Vehículo:
Marca: {row['marca']}
Línea: {row['linea']}
Versión: {row['version']}
Modelo: {row['modelo']}
Edad del vehículo: {row['vehicle_age']}

Versión de los hechos:
{row['version_hechos']}

Piezas inspeccionadas:
{row['parts_text']}

Número total de piezas: {row['piezas_totales']}
Número de piezas con descripción válida: {row['valid_parts']}
""".strip()

    return query