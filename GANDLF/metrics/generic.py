import torch
from torchmetrics import F1, Precision, Recall, IoU, Accuracy
from GANDLF.utils.tensor import one_hot


def generic_function_output_with_check(predicted_classes, label, metric_function):
    # empty batches are seen when training with opacus
    if (predicted_classes.nelement() == 0) and (label.nelement() == 0):
        return torch.zeros((1), device=predicted_classes.device)
    elif torch.min(predicted_classes) < 0:
        print(
            "WARNING: Negative values detected in prediction, cannot compute torchmetrics calculations."
        )
        return torch.zeros((1), device=predicted_classes.device)
    else:
        predicted_new = torch.clamp(
            predicted_classes.cpu().int(), max=metric_function.num_classes - 1
        )
        predicted_new = predicted_new.reshape(label.shape)
        return metric_function(predicted_new, label.cpu().int())


def generic_torchmetrics_score(output, label, metric_class, metric_key, params):
    num_classes = params["model"]["num_classes"]
    predicted_classes = output
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    elif params["problem_type"] == "segmentation":
        label = one_hot(label, params["model"]["class_list"])
    else:
        params["metrics"][metric_key]["multi_class"] = False
        params["metrics"][metric_key]["mdmc_average"] = None
    metric_function = metric_class(
        average=params["metrics"][metric_key]["average"],
        num_classes=num_classes,
        multiclass=params["metrics"][metric_key]["multi_class"],
        mdmc_average=params["metrics"][metric_key]["mdmc_average"],
        threshold=params["metrics"][metric_key]["threshold"],
    )

    return generic_function_output_with_check(
        predicted_classes.cpu().int(), label.cpu().int(), metric_function
    )


def recall_score(output, label, params):
    return generic_torchmetrics_score(output, label, Recall, "recall", params)


def precision_score(output, label, params):
    return generic_torchmetrics_score(output, label, Precision, "precision", params)


def f1_score(output, label, params):
    return generic_torchmetrics_score(output, label, F1, "f1", params)


def accuracy(output, label, params):
    return generic_torchmetrics_score(output, label, Accuracy, "accuracy", params)


def iou_score(output, label, params):
    num_classes = params["model"]["num_classes"]
    predicted_classes = output
    if params["problem_type"] == "classification":
        predicted_classes = torch.argmax(output, 1)
    elif params["problem_type"] == "segmentation":
        label = one_hot(label, params["model"]["class_list"])

    recall = IoU(
        reduction=params["metrics"]["iou"]["reduction"],
        num_classes=num_classes,
        threshold=params["metrics"]["iou"]["threshold"],
    )

    return generic_function_output_with_check(
        predicted_classes.cpu().int(), label.cpu().int(), recall
    )
