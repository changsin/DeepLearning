import numpy as np

import FSDL.plate_recognizer.training.metrics as metrics

from FSDL.plate_recognizer.data.base_data_module import DataType
from FSDL.plate_recognizer.utils.logger import get_logger

logger = get_logger(__name__)


# run predictions many times to get the distributions
def sample_predictions(model, X, iterations=50):
    """
    Need to change 3x2x4 to 2x3x4.
        (iterations)x(samples)x(coordinates)
    to
        (samples)x(iterations)x(coordinates) 
    """
    predicted = []
    for _ in range(iterations):
        predicted.append(model(X).numpy())

    predicted = np.array(predicted)
    # predicted = np.concatenate(predicted, axis=1)

    # predicted = np.array([model_prob.predict(np.expand_dims(X_test[1], [0])) for i in range(iterations)])
    # predicted = np.concatenate(predicted, axis=1)
    reshaped = np.array([predicted[:, column] for column in range(0, predicted.shape[1])])

    return reshaped

def average_sample_preds(y_sample_preds):
    # averages out sample predictions (input data)x(samples)x(prediction points = 4)
    # into (input data)x(averaged prediction points = 4)
    averages = []
    for y_pred in y_sample_preds:
        averages.append([np.mean(y_pred[:, i]) for i in range(y_pred.shape[1])])
    return np.array(averages)

def predict_on_cluster(model, X_test, y_test, iterations=50, iou_threshold=0.5):
    test_accuracy = 0
    test_loss, test_accuracy = model.evaluate(X_test, y_test, steps=1)

    y_preds = sample_predictions(model, X_test, iterations=iterations)
    preds_avg = average_sample_preds(y_preds)

#    m_ap = metrics.calculate_map(y_test, preds_avg, iou_threshold=iou_threshold)
    m_ap = metrics.mean_average_precision(y_test, preds_avg)
    stds = np.mean(np.std(y_preds, axis=1), axis=1)

    return y_preds, m_ap, np.mean(stds, axis=0), test_accuracy

def predict_on_models(dataset, bins, models, iterations=50, iou_threshold=0.5):
    stats = []
    for model in models:
        cluster_stats = []
        for cluster_id in bins:
            X_test, Y_test = dataset.get_data(data_type=DataType.Test, cluster_id=cluster_id)
            y_preds, m_ap, std, accuracy = predict_on_cluster(model,
                                                              X_test, Y_test,
                                                              iterations=iterations,
                                                              iou_threshold=iou_threshold)
            logger.info("{} mAP: {:0.2f} std: {:0.2f} acc: {:0.2f}".format(cluster_id,
                                                                    m_ap,
                                                                    std,
                                                                    accuracy))
            cluster_stats.append([np.round(m_ap['avg_prec'], 3), np.round(std, 3), np.round(accuracy, 3)])
        stats.append(cluster_stats)

    return np.array(stats)


def get_prediction_distributions():
    min_id = np.argmin(mean_stds)
    max_id = np.argmax(mean_stds)

    print("min:", min_id, mean_stds[min_id])
    print("max:", max_id, mean_stds[max_id])

    [print(f"{id} pred std: ", stds[id], f"mean_std:  {mean_stds[id]})") for id in range(len(X_test))]
