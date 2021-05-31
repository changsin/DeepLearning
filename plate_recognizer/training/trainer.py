
import logging
import wandb

from FSDL.plate_recognizer.data.base_data_module import DataType

logger = logging.getLogger(__name__)

class Trainer():
    def __init__(self, model):
        self.model = model
    
        wandb.init(project="Kaggle-license-plates",
                config={
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "dataset": "Kaggle-license-plates",
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

        logger.info("Test results \n Loss:", test_loss, '\n Accuracy', test_accuracy)

        y_preds = self.sample_predictions(self.model, X_test, iterations=1)

        if is_plot_predictions:
            plot_predictions(X_test, Y_test, y_preds)

        # averaged_predictions = average_sample_preds(y_preds)
        # y_test = np.array([to_rect(y*IMAGE_SIZE) for y in y_test])
        # rectified_predictions = np.array([to_rect(y*IMAGE_SIZE) for y in averaged_predictions])

        # # print(rectified_predictions)
        # m_ap = calculate_map(y_test*IMAGE_SIZE, rectified_predictions*IMAGE_SIZE)
        return model
    
    # run predictions many times to get the distributions
    def sample_predictions(self, X, iterations=100):
        predicted = []
        for _ in range(iterations):
            predicted.append(self.model(X).numpy())

        predicted = np.array(predicted)
        # predicted = np.concatenate(predicted, axis=1)

        # predicted = np.array([model_prob.predict(np.expand_dims(X_test[1], [0])) for i in range(iterations)])
        # predicted = np.concatenate(predicted, axis=1)
        reshaped = np.array([predicted[:, column] for column in range(0, predicted.shape[1])])

        return reshaped
