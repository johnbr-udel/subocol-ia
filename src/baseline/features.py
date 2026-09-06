from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix


def create_feature_transformers():
    narrative_vectorizer = TfidfVectorizer()
    parts_vectorizer = TfidfVectorizer()

    brand_encoder = OneHotEncoder(
        handle_unknown="ignore"
    )

    numeric_scaler = StandardScaler()

    return (
        narrative_vectorizer,
        parts_vectorizer,
        brand_encoder,
        numeric_scaler
    )


def fit_transform_train_features(
    train_df,
    narrative_vectorizer,
    parts_vectorizer,
    brand_encoder,
    numeric_scaler
):
    # TF-IDF from accident narrative
    X_narrative = narrative_vectorizer.fit_transform(
        train_df["version_hechos"]
    )

    # TF-IDF from inspected parts
    X_parts = parts_vectorizer.fit_transform(
        train_df["parts_text"]
    )

    # Vehicle brand
    X_brand = brand_encoder.fit_transform(
        train_df[["marca"]]
    )

    numeric_columns = [
        "vehicle_age",
        "piezas_totales",
        "valid_parts"
    ]

    X_numeric = numeric_scaler.fit_transform(
        train_df[numeric_columns]
    )

    X_numeric = csr_matrix(X_numeric)

    X_train = hstack([
        X_narrative,
        X_parts,
        X_brand,
        X_numeric
    ]).tocsr()

    return X_train


def transform_features(
    df,
    narrative_vectorizer,
    parts_vectorizer,
    brand_encoder,
    numeric_scaler
):
    X_narrative = narrative_vectorizer.transform(
        df["version_hechos"]
    )

    X_parts = parts_vectorizer.transform(
        df["parts_text"]
    )

    X_brand = brand_encoder.transform(
        df[["marca"]]
    )

    numeric_columns = [
        "vehicle_age",
        "piezas_totales",
        "valid_parts"
    ]

    X_numeric = numeric_scaler.transform(
        df[numeric_columns]
    )

    X_numeric = csr_matrix(X_numeric)

    X = hstack([
        X_narrative,
        X_parts,
        X_brand,
        X_numeric
    ]).tocsr()

    return X