import argparse
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

model_path = "./files/model/model.h5"
train_path = "./files/data/train.csv"
dev_path = "./files/data/val.csv"
test_path = "./files/data/test.csv"
output_path = "./files/data/output.csv"

tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")

def tokenize(examples):
    return tokenizer(examples["response"], truncation=True, max_length=128, padding="max_length")

# --- R2 Metric ---
class R2Score(tf.keras.metrics.Metric):
    def __init__(self, name="r2_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sse = self.add_weight(name="sse", initializer="zeros")
        self.sst = self.add_weight(name="sst", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        self.sse.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
        self.sst.assign_add(tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))))

    def result(self):
        return 1.0 - (self.sse / (self.sst + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.sse.assign(0.0)
        self.sst.assign(0.0)

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_states(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

def create_regression_model(input_dim):
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")

    x = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=128)(inputs)

    x = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    output = tf.keras.layers.Dense(1, activation='sigmoid', name="deception_score")(x)

    return tf.keras.Model(inputs=inputs, outputs=output)


def train(model_path=model_path, train_path=train_path, dev_path=dev_path):
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    def prepare_labels(example):
        return {
            "deception_score": float(example["Deception_Score"])
        }



    hf_dataset = hf_dataset.map(prepare_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns="input_ids",
        label_cols="deception_score",
        batch_size=16,
        shuffle=True
    )

    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns="input_ids",
        label_cols="deception_score",
        batch_size=16
    )


    model = create_regression_model(input_dim=tokenizer.vocab_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            R2Score()
        ]
    )

    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=dev_dataset,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True)
        ]
    )


    def plot_training_history(history):
        epochs = range(1, len(history.history["loss"]) + 1)

        plt.plot(epochs, history.history["val_loss"], label="Val MSE")
        plt.plot(epochs, history.history["val_r2_score"], label="Val R²")
        plt.xlabel("Epochs")
        plt.ylabel("Metric")
        plt.title("Regression Performance")
        plt.legend()
        plt.show()


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
