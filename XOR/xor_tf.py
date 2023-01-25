import tensorflow as tf

# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim=2, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Prepare the input and target data
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

# Train the model
model.fit(x, y, epochs=1000, batch_size=1)

# Print the final output of the model
print(model.predict(x))
