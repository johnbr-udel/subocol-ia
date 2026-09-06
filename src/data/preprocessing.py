import pandas as pd


def build_claim_dataset(df):
    claims = (
        df.groupby("numero_aviso")
        .agg(
            fecha_creacion=("fecha_creacion", "first"),
            tipo_carroceria=("tipo_carroceria", "first"),
            marca=("marca", "first"),
            linea=("linea", "first"),
            version=("version", "first"),
            modelo=("modelo", "first"),
            piezas_totales=("piezas_totales", "first"),
            piezas_cambio=("piezas_cambio", "first"),
            version_hechos=("version_hechos", "first"),
            estado_aviso=("estado_aviso", "first"),
            valid_parts=("nombre_irs", "count")
        )
    )

    parts_by_claim = (
        df.groupby("numero_aviso")["nombre_irs"]
        .apply(lambda x: x.dropna().tolist())
    )

    claims["parts"] = parts_by_claim
    claims = claims.reset_index()

    claims["fecha_creacion"] = pd.to_datetime(
        claims["fecha_creacion"],
        errors="coerce"
    )

    claims["parts_text"] = claims["parts"].apply(
        lambda parts: " ".join(parts)
    )

    claims["vehicle_age"] = (
        claims["fecha_creacion"].dt.year
        - claims["modelo"]
    ).clip(lower=0)

    return claims