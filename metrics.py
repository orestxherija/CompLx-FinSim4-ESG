import numpy
import sklearn.metrics
import transformers


def compute_classification_metrics(p: transformers.EvalPrediction):
    y_pred = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    y_true = p.label_ids
    y_pred = numpy.argmax(y_pred, axis=1)

    recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred)
    precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred)
    f1 = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_pred)
    _, fp, fn, _ = sklearn.metrics.confusion_matrix(y_true=y_pred, y_pred=y_pred).ravel()
    return {
        "accuracy": (y_pred == y_true).astype(numpy.float32).mean().item(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp": fp,
        "fn": fn
    }
