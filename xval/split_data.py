import json
import random
from datasets import DatasetDict, Features, Value

# Function to split data
def split_data(data, train_ratio, val_ratio):
    # Shuffle the data randomly for fair distribution
    random.shuffle(data)
    
    # Calculate split indices
    total_samples = len(data)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    # Split the data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data

# Read the data from the file
#with open('planet_sims/.f6f5e64a-772e-46ab-abac-db75b09ad73c', 'r') as file:
#    data = json.load(file)
ds = DatasetDict.from_text('xval/planet_sims/.f6f5e64a-772e-46ab-abac-db75b09ad73c', 
                           features=Features({'input': Value('string'), 'output': Value('string')}))

# Assume data is a list of samples
train_data, val_data, test_data = split_data(data, train_ratio=0.8, val_ratio=0.1)

# Optionally, save the split data back to new JSON files
with open('train_data.json', 'w') as f:
    json.dump(train_data, f)
with open('val_data.json', 'w') as f:
    json.dump(val_data, f)
with open('test_data.json', 'w') as f:
    json.dump(test_data, f)