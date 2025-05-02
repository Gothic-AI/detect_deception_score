import argparse
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy

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

def create_multitask_model(input_dim):
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    x = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=128)(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    deception_score = tf.keras.layers.Dense(1, activation='sigmoid', name="deception_score")(x)
    deception_class = tf.keras.layers.Dense(1, activation='sigmoid', name="deception")(x)

    return tf.keras.Model(inputs=inputs, outputs={
        "deception_score": deception_score,
        "deception": deception_class
    })

def train(model_path=model_path, train_path=train_path, dev_path=dev_path):
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    def prepare_labels(example):
        return {
            "deception_score": float(example["Deception_Score"]),
            "deception": float(example["Deception"])
        }


    hf_dataset = hf_dataset.map(prepare_labels)
    hf_dataset = hf_dataset.map(tokenize, batched=True)

    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns="input_ids",
        label_cols=["deception_score", "deception"],
        batch_size=16,
        shuffle=True)

    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns="input_ids",
        label_cols=["deception_score", "deception"],
        batch_size=16)


    model = create_multitask_model(input_dim=tokenizer.vocab_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss={
            "deception_score": tf.keras.losses.MeanSquaredError(),
            "deception": tf.keras.losses.BinaryCrossentropy()
        },
        metrics={
            "deception_score": [tf.keras.metrics.MeanAbsoluteError(), R2Score()],
            "deception": [tf.keras.metrics.BinaryAccuracy()]
        }
    )

    model.fit(
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

def predict(model_path=model_path, input_path=test_path):
    model = tf.keras.models.load_model(model_path, custom_objects={"R2Score": R2Score})

    df = pandas.read_csv(input_path)

    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    tf_dataset = hf_dataset.to_tf_dataset(
        columns="input_ids",
        batch_size=16)

    predictions = model.predict(tf_dataset)
    df["Deception_Score"] = predictions["deception_score"].flatten()
    df["Deception"] = (predictions["deception"].flatten() > 0.5).astype(int)

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    globals()[args.command]()
