import pickle, numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Configuration
SEQ_LEN = 20
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

print("Loading notes...")
try:
    notes = pickle.load(open("notes.pkl", "rb"))
    print(f"Loaded {len(notes)} notes from notes.pkl")
except FileNotFoundError:
    print("WARNING: notes.pkl not found. Using sample data.")
    notes = ['C4', 'D4', 'E4'] * 100

# Create vocabulary
pitchnames = sorted(set(notes))
n_vocab = len(pitchnames)
note_to_int = dict((n, i) for i, n in enumerate(pitchnames))
int_to_note = dict((i, n) for n, i in note_to_int.items())

print(f"Vocabulary size: {n_vocab} unique notes")
print(f"Total notes: {len(notes)}")

# Prepare training data
network_input = []
network_output = []

for i in range(len(notes) - SEQ_LEN):
    seq = notes[i:i + SEQ_LEN]
    out = notes[i + SEQ_LEN]
    network_input.append([note_to_int[n] for n in seq])
    network_output.append(note_to_int[out])

if len(network_input) == 0:
    print("ERROR: Need more MIDI data. Not enough notes to create sequences.")
    exit(1)

print(f"Created {len(network_input)} training sequences")

# Reshape input data
X = np.reshape(network_input, (len(network_input), SEQ_LEN, 1)) / float(n_vocab)
Y = np.array(network_output)

# Split into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=VALIDATION_SPLIT, random_state=42
)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Build model
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(n_vocab, activation='softmax'))

# Compile model
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print("\nModel Architecture:")
model.summary()

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Callbacks
checkpoint = ModelCheckpoint(
    "model/music_model_checkpoint.keras",
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    verbose=1,
    restore_best_weights=True
)

# Train model
print("\nStarting training...")
history = model.fit(
    X_train, Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, Y_val),
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# Save final model
model.save("model/music_model.keras")
print("\nTraining complete! Model saved to model/music_model.keras")

# Save vocabulary
with open("model/note_mapping.pkl", "wb") as f:
    pickle.dump((note_to_int, int_to_note), f)
print("Note mapping saved to model/note_mapping.pkl")
