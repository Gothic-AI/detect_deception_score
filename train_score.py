import argparse
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import datetime
import os

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)


model_path = "./files/model/model.keras"
train_path = "./files/drakula_data/train.csv"
dev_path = "./files/drakula_data/val.csv"
test_path = "./files/drakula_data/test.csv"
output_path = "./files/drakula_data/output.csv"

tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")

def tokenize(examples):
    return tokenizer(examples["Response"], truncation=True, max_length=128, padding="max_length")

# --- R2 Metric ---
class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name="r2_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sse = self.add_weight(name="sse", initializer="zeros")
        self.sst = self.add_weight(name="sst", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.mean_y = self.add_weight(name="mean_y", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Update running mean
        batch_count = tf.cast(tf.shape(y_true)[0], tf.float32)
        total_count = self.count + batch_count
        new_mean = (self.mean_y * self.count + tf.reduce_sum(y_true)) / total_count

        # Compute batch SSE and SST
        batch_sse = tf.reduce_sum(tf.square(y_true - y_pred))
        batch_sst = tf.reduce_sum(tf.square(y_true - new_mean))

        # Update state
        self.sse.assign_add(batch_sse)
        self.sst.assign_add(batch_sst)
        self.count.assign(total_count)
        self.mean_y.assign(new_mean)

    def result(self):
        return 1.0 - (self.sse / (self.sst + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.sse.assign(0.0)
        self.sst.assign(0.0)
        self.count.assign(0.0)
        self.mean_y.assign(0.0)



def create_regression_model(input_dim):
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")

    x = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=256)(inputs)

    x = tf.keras.layers.Conv1D(filters=128, kernel_size=128, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(filters=128, kernel_size=64, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    output = tf.keras.layers.Dense(1, activation='sigmoid', name="deception_score")(x)

    return tf.keras.Model(inputs=inputs, outputs=output)


def train(model_path=model_path, train_path=train_path, dev_path=dev_path):
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    def prepare_labels(example):
        return {
            "deception_score": float(example["Deception Score"])
        }

    # Shuffle train and validation sets
    hf_dataset["train"] = hf_dataset["train"].shuffle(seed=42)
    hf_dataset["validation"] = hf_dataset["validation"].shuffle(seed=42)

    hf_dataset = hf_dataset.map(prepare_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns="input_ids",
        label_cols="deception_score",
        batch_size=8,
        shuffle=True
    )

    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns="input_ids",
        label_cols="deception_score",
        batch_size=8
    )


    model = create_regression_model(input_dim=tokenizer.vocab_size)

    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(0.001), #'adam',
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mean_absolute_error'),
            R2Score(name='r2_score')
        ]
    )


    history = model.fit(
        train_dataset,
        epochs=2,
        validation_data=dev_dataset,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True),
            tensorboard_cb
        ]
    )


    # def plot_training_history(history, save_path="plots/regression_performance.png"):
    #     epochs = range(1, len(history.history["loss"]) + 1)

    #     plt.plot(epochs, history.history["val_loss"], label="Val MSE")
    #     plt.plot(epochs, history.history["val_r2_score"], label="Val R²")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Metric")
    #     plt.title("Regression Performance")
    #     plt.legend()
    #     plt.show()

    def plot_training_history(history, save_path="plots/regression_performance.png"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        epochs = range(1, len(history.history["loss"]) + 1)

        plt.figure(figsize=(8, 6))
        plt.plot(epochs, history.history["val_loss"], label="Val MSE")
        plt.plot(epochs, history.history["val_r2_score"], label="Val R²")
        plt.xlabel("Epochs")
        plt.ylabel("Metric")
        plt.title("Regression Performance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(save_path)  # Save the plot
        # plt.show()              # Optionally show it


    plot_training_history(history)


def predict(model_path=model_path, input_path=test_path):
    # Load model with custom R2Score metric
    model = tf.keras.models.load_model(model_path, custom_objects={"R2Score": R2Score})

    # Load and tokenize test data
    df = pandas.read_csv(input_path)
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    # Create tf.Dataset
    tf_dataset = hf_dataset.to_tf_dataset(
        columns="input_ids",
        batch_size=16
    )

    # Predict regression score
    predictions = model.predict(tf_dataset)
    df["Deception_Score"] = predictions.flatten()

    # Save predictions
    df.to_csv(output_path, index=False)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    globals()[args.command]()
