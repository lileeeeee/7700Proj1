import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

from mlp import Layer, MultilayerPerceptron
from mlp import Relu, Softmax, CrossEntropy

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:

    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    for i, lab in enumerate(labels):
        encoded[i, lab] = 1.0
    return encoded

def select_samples_per_class(x_test, y_test, y_pred):
    selected_images = []
    selected_true_labels = []
    selected_pred_labels = []
    seen_classes = set()

    for i in range(len(y_test)):
        label = y_test[i]
        if label not in seen_classes:
            selected_images.append(x_test[i].reshape(28, 28))  # Reshape to 28x28
            selected_true_labels.append(label)
            selected_pred_labels.append(y_pred[i])
            seen_classes.add(label)

        if len(seen_classes) == 10:
            break

    return selected_images, selected_true_labels, selected_pred_labels

def plot_samples(images, true_labels, pred_labels):
    """
    Plots the selected samples along with their true and predicted labels.
    """
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target

    X = X.astype(np.float32)
    y = y.astype(np.int64)


    x_train, x_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    x_train /= 255.0
    x_test  /= 255.0


    y_train_oh = one_hot_encode(y_train, num_classes=10)
    y_test_oh  = one_hot_encode(y_test,  num_classes=10)

    num_train = int(x_train.shape[0] * 0.8)  # 80%
    x_train_split = x_train[:num_train]
    y_train_split = y_train_oh[:num_train]
    x_val_split   = x_train[num_train:]
    y_val_split   = y_train_oh[num_train:]

    layer1 = Layer(784, 256, Relu())
    layer2 = Layer(256, 128, Relu())
    layer3 = Layer(128, 64, Relu(), 0.1)
    layer_out = Layer(64, 10, Softmax())

    model = MultilayerPerceptron((layer1, layer2, layer3, layer_out))


    loss_func = CrossEntropy()


    print("Start training MLP on MNIST ...")
    train_losses, val_losses = model.train(
        x_train_split,
        y_train_split,
        x_val_split,
        y_val_split,
        loss_func,
        learning_rate=1e-3,
        batch_size=64,
        epochs=50,
        rmsprop=False
    )

    y_pred_test = model.forward(x_test, training=False)
    pred_labels = np.argmax(y_pred_test, axis=1)

    correct = (pred_labels == y_test).sum()
    total = x_test.shape[0]
    test_acc = correct / total

    "Plotting the selected samples"
    images, true_labels, pred_labels = select_samples_per_class(x_test, y_test, pred_labels)
    plot_samples(images, true_labels, pred_labels)

    "Plotting the loss"
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MNIST MLP: Training & Validation Loss')
    plt.legend()
    plt.savefig('mnist_loss.png')

    print(f"\nFinished training, test accuracy = {test_acc:.4f}")

if __name__=="__main__":
    main()