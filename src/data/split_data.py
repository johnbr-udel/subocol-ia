from sklearn.model_selection import train_test_split


def split_claim_data(df):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["estado_aviso"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["estado_aviso"]
    )

    return train_df, val_df, test_df