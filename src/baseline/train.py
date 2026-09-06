from sklearn.linear_model import LogisticRegression


def train_logistic_regression(
    X_train,
    y_train,
    C=1,
    class_weight=None
):
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model