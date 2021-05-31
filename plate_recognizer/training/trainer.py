
import wandb
import numpy as np
import FSDL.plate_recognizer.training.metrics as metrics
import FSDL.plate_recognizer.training.predictor as predictor

from FSDL.plate_recognizer.data.base_data_module import DataType
from FSDL.plate_recognizer.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer():
    def __init__(self, model, name="Kaggle-license-plates"):
        self.model = model
    
        wandb.init(project=name,
                config={
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "dataset": name,
                })

    def train(self, dataset, epochs=50, batch_size=16, is_plot_predictions=False):
        X_train, Y_train = dataset.get_data(DataType.Train)
        X_val, Y_val = dataset.get_data(DataType.Val)
        X_test, Y_test = dataset.get_data(DataType.Test)

        train_history = self.model.fit(x=X_train, y=Y_train,
                                    validation_data=(X_val, Y_val),
                                    epochs=epochs, batch_size=batch_size, verbose=1,
                                    callbacks=[wandb.keras.WandbCallback(data_type="image",
                                    save_model=False)])
        # Test
        scores = self.model.evaluate(X_test, Y_test, verbose=0)
        logger.info("Score : %.2f%%" % (scores[1]*100))

        test_loss, test_accuracy = self.model.evaluate(X_test, Y_test, steps=int(100))

        logger.info("Test results \n Loss: {}\n Accuracy: {}".format(test_loss, test_accuracy))

        y_preds = predictor.sample_predictions(X_test, iterations=1)

        if is_plot_predictions:
            plot_predictions(X_test, Y_test, y_preds)

        # averaged_predictions = average_sample_preds(y_preds)
        # y_test = np.array([to_rect(y*IMAGE_SIZE) for y in y_test])
        # rectified_predictions = np.array([to_rect(y*IMAGE_SIZE) for y in averaged_predictions])

        # # print(rectified_predictions)
        # m_ap = calculate_map(y_test*IMAGE_SIZE, rectified_predictions*IMAGE_SIZE)
        return self.model
