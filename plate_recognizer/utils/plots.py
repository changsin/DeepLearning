import matplotlib.patches as patches
import numpy as np

from matplotlib import pyplot as plt

IMAGE_SIZE = 224


def create_patch_rectangle(y, color):
  # width = y[2] - y[0]
  # height = y[3] - y[1]
  # return patches.Rectangle((y[0], y[1]),
  #                           width, height,
  #                           edgecolor=color, fill=False)
  # # in yolov5
  width = int(y[2])
  height = int(y[3])
  return patches.Rectangle((int(y[0] - width/2), int(y[1] - height/2)),
                           width, height,
                           edgecolor=color, fill=False)
def plot_images(X, y, limit=10):
  fig = plt.figure(figsize=(20,40))

  # The number of images for plotting is limited to 50
  end_id = len(y) if len(y) < limit else limit

  for i in range(0, end_id):
    axis = fig.add_subplot(10, 5, i+1)
    plt.axis('off')
    image = X[i]

    rect_ori = create_patch_rectangle(y[i]*IMAGE_SIZE, (0, 255/255, 0))
    axis.add_patch(rect_ori)
    plt.imshow(np.clip(image, 0, 1))

def plot_pca(X_train_pca, y_train):
    # plot the scatter plot along the way
    plt.figure(1)
    plt.clf()

    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap="Paired")
    plt.colorbar()


def plot_cluster_histogram(X_clusters, K):
    histo_x, bins = np.histogram(X_clusters.labels_, bins=range(K + 1))
    plt.bar(bins[:-1], histo_x, align='center')

def plot_pca_clusters(X_train_pca, kmeans):
    # kmeans, X_train_pca_clusters = get_clusters(X_train_pca)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.figure(2)
    # plt.clf()
    plt.imshow(Z, interpolation="nearest",
                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                cmap=plt.cm.Paired, aspect="auto", origin="lower")

    plt.plot(X_train_pca[:, 0], X_train_pca[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_

    markers = ["o", "1", "2", "3", "4"]
    for id in range(len(centroids)):
        c = centroids[id]
        plt.scatter(c[0], c[1], marker=markers[id], s=169, linewidths=3,
                    color="w", zorder=10)

    # plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
    #             color="w", zorder=10)
    # https://matplotlib.org/2.0.2/api/markers_api.html#module-matplotlib.markers
    plt.title("K-means clustering on the PCA-reduced data\n"
                "Centroids 0-o, 1-down, 2-up, 3-left, 4-right tri")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def plot_data_in_clusters(X_train_pca, kmeans, idx):
    # kmeans, X_train_pca_clusters = get_clusters(X_train_pca)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X_train_pca[:, 0].min() - 3, X_train_pca[:, 0].max() + 3
    y_min, y_max = X_train_pca[:, 1].min() - 3, X_train_pca[:, 1].max() + 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.figure(2)
    # plt.clf()
    plt.imshow(Z, interpolation="nearest",
                extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                cmap=plt.cm.Paired, aspect="auto", origin="lower")

    plt.plot(X_train_pca[:, 0], X_train_pca[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    # centroids = kmeans.cluster_centers_

    # markers = ["o", "1", "2", "3", "4"]
    # for id in range(len(centroids)):
    #   c = centroids[id]
    #   plt.scatter(c[0], c[1], marker=markers[id], s=169, linewidths=3,
    #               color="w", zorder=10)
    for id in idx:
        plt.scatter(X_train_pca[id, 0], X_train_pca[id, 1], marker="x",
                    s=169, linewidths=3,
                    color="w", zorder=10)

    # plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
    #             color="w", zorder=10)
    # https://matplotlib.org/2.0.2/api/markers_api.html#module-matplotlib.markers
    plt.title("K-means clustering on the PCA-reduced data\n"
                "Centroids 0-o, 1-down, 2-up, 3-left, 4-right tri")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def plot_predictions(X, y_gt, y_preds):
  fig = plt.figure(figsize=(20,40))

  # The number of images for plotting is limited to 50
  end_id = len(y_gt) if len(y_gt) < 50 else 50

  y_preds_avg = average_sample_preds(y_preds)
  stds = np.std(y_preds, axis=1)
  mean_stds = np.mean(stds, axis=1)

  for i in range(0, end_id):
    axis = fig.add_subplot(10, 5, i+1)
    plt.axis('off')
    image = X[i]

    rect_ori = create_patch_rectangle(y_gt[i]*IMAGE_SIZE, (0, 255/255, 0))
    axis.add_patch(rect_ori)

    # for each test image, there could be multiple predictions
    for y_pred in y_preds[i]:
      rect_pred = create_patch_rectangle(y_pred*IMAGE_SIZE, (255/255, 0, 0))
      axis.add_patch(rect_pred)

    iou = bb_iou(to_rect(y_preds_avg[i]*IMAGE_SIZE), to_rect(y_gt[i]*IMAGE_SIZE))
    plt.title("IOU: {:0.2f} std: {:0.2f}".format(iou, mean_stds[i]))
    # plt.title("mean std: {:0.2f}".format(mean_stds[sample_ids[i]]))
    plt.imshow(np.clip(image, 0, 1))

def plot_stats(stats, types, titles):
  x_bar = np.arange(K)
  bar_width = .35

  fig = plt.figure(figsize=(20,40))

  for i, stat in zip(range(len(stats)), stats):
    ax = fig.add_subplot(10, 4, i+1)
    # plt.axis('off')

    for t in types:
      if "mAP" == t:
        rects1 = ax.bar(x_bar - bar_width/3, stat[:, 0], label="mAP")
      if "std" == t:
        rects2 = ax.bar(x_bar + bar_width/3, stat[:, 1], label="std")
      if "accuracy" == t:
        rects3 = ax.bar(x_bar + bar_width/3, stat[:, 2], label="accuracy")

    ax.set_xticks(x_bar)
    ax.set_xticklabels(x_bar)
    ax.set_title(titles[i])

    ax.legend()

  fig.tight_layout()