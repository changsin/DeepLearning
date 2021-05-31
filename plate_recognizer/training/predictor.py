import numpy as np

import FSDL.plate_recognizer.training.predictor as predictor

from FSDL.plate_recognizer.data.base_data_module import DataType
from FSDL.plate_recognizer.utils.logger import get_logger

logger = get_logger(__name__)


# run predictions many times to get the distributions
def sample_predictions(model, X, iterations=50):
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

def predict_on_cluster(model, X_test, y_test, is_plot_predictions=False, iterations=50):
    test_accuracy = 0
    test_loss, test_accuracy = model.evaluate(X_test, y_test, steps=1)
    y_preds = sample_predictions(X_test, iterations=iterations)

    preds_avg = average_sample_preds(y_preds)
    rectified_y_test = np.array([to_rect(y*IMAGE_SIZE) for y in y_test])
    rectified_predictions = np.array([to_rect(y*IMAGE_SIZE) for y in preds_avg])

    m_ap = metrics.calculate_map(rectified_y_test*IMAGE_SIZE, rectified_predictions*IMAGE_SIZE)
    stds = np.mean(np.std(y_preds, axis=1), axis=1)

    return y_preds, m_ap, np.mean(stds, axis=0), test_accuracy

def predict_on_models(X_d, y_d, bins, models):
    stats = []
    for model in models:
        cluster_stats = []
        for clst_id in bins:
            y_preds, m_ap, std, accuracy = predict_on_cluster(model, X_d[clst_id], y_d[clst_id])
            logger.info("{} mAP: {:0.2f} std: {:0.2f} acc: {:0.2f}".format(clst_id,
                                                                    m_ap['avg_prec'],
                                                                    std,
                                                                    accuracy))
            cluster_stats.append([np.round(m_ap['avg_prec'], 3), np.round(std, 3), np.round(accuracy, 3)])

            stats.append(cluster_stats)

    return np.array(stats)
