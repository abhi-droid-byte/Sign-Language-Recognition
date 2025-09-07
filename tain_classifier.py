import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Function to pad sequences
def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', value=0.0):
    if maxlen is None:
        maxlen = max(len(x) for x in sequences)
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = np.full((len(sequences), maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        trunc = np.asarray(s[:maxlen], dtype=dtype)
        x[idx, :len(trunc)] = trunc
    return x

# Pad data to ensure consistency
data = pad_sequences(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate and print the accuracy score
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
