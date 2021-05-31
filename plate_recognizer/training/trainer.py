
import logging
import wandb



logger = logging.getLogger(__name__)

class Trainer():
    def __init__(self, model):
        self.model = model
    
        wandb.init(project="UFPR-cnn",
                config={
                    "batch_size": 16,
                    "learning_rate": 0.01,
                    "dataset": "UFPR-cnn",
                })

    def train(self, dataset, epochs=50, batch_size=16, is_plot_predictions=False):
        train_history = model.fit(x=dataset.get_x_train(), y=dataset.get_y_train(),
                                    validation_data=(dataset.get_x_val(), dataset.get_y_val()),
                                    epochs=epochs, batch_size=batch_size, verbose=1,
                                    callbacks=[wandb.keras.WandbCallback(data_type="image",
                                    save_model=False)])
        # Test
        scores = model.evaluate(X_test, y_test, verbose=0)
        logger.info("Score : %.2f%%" % (scores[1]*100))

        test_loss, test_accuracy = model.evaluate(dataset.get_x_test(), dataset.get_y_test(), steps=int(100))

        logger.info("Test results \n Loss:",test_loss,'\n Accuracy',test_accuracy)

        y_preds = self.sample_predictions(model, dataset.get_x_test(), iterations=1)
        # y_preds = model.predict(X_test)

        # # TODO:
        # # Hack to fix erroneous predictions
        # y_preds = fix_predictions(y_preds)
        if is_plot_predictions:
            plot_predictions(dataset.get_x_test(), dataset.get_y_test(), y_preds)

        # averaged_predictions = average_sample_preds(y_preds)
        # y_test = np.array([to_rect(y*IMAGE_SIZE) for y in y_test])
        # rectified_predictions = np.array([to_rect(y*IMAGE_SIZE) for y in averaged_predictions])

        # # print(rectified_predictions)
        # m_ap = calculate_map(y_test*IMAGE_SIZE, rectified_predictions*IMAGE_SIZE)
        return model
    
    # run predictions many times to get the distributions
    def sample_predictions(self, model, X, iterations=100):
        predicted = []
        for _ in range(iterations):
            predicted.append(model(X).numpy())

        predicted = np.array(predicted)
        # predicted = np.concatenate(predicted, axis=1)

        # predicted = np.array([model_prob.predict(np.expand_dims(X_test[1], [0])) for i in range(iterations)])
        # predicted = np.concatenate(predicted, axis=1)
        reshaped = np.array([predicted[:, column] for column in range(0, predicted.shape[1])])

        return reshaped
