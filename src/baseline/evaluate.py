from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score
)


def get_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["ENTREGADO", "OBJETADO"]
    )

    return cm


def evaluate_predictions(y_true, y_pred):
    precision = precision_score(
        y_true,
        y_pred,
        pos_label="OBJETADO"
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="OBJETADO"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="OBJETADO"
    )

    f2 = fbeta_score(
        y_true,
        y_pred,
        beta=2,
        pos_label="OBJETADO"
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2
    }