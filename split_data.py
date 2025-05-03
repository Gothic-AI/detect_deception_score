import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full Q&A dataset
df = pd.read_csv("./files/drakula_data/Dracula_Deception_Dataset_With_Responses.csv")  # Replace with your actual file path

# First split: 80% train, 20% temp
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Second split: 10% val, 10% test from the 20% temp
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save the splits
train_df.to_csv("./files/drakula_data/train.csv", index=False)
val_df.to_csv("./files/drakula_data/val.csv", index=False)
test_df.to_csv("./files/drakula_data/test.csv", index=False)

print("Splits saved:")
print(f"Train: {len(train_df)} samples")
print(f"Validation: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")
