import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
X, y = load_iris(return_X_y=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Find centers using K-means clustering
num_centers = 3
kmeans = KMeans(n_clusters=num_centers, random_state=42).fit(X_train)
centers = kmeans.cluster_centers_

# Define RBF layer
class RBFLayer(keras.layers.Layer):
    def __init__(self, num_centers):
        super(RBFLayer, self).__init__()
        self.num_centers = num_centers
        self.centers = self.add_weight(name='centers', shape=(self.num_centers, X_train.shape[1]), initializer='uniform', trainable=True)
        self.widths = self.add_weight(name='widths', shape=(self.num_centers,), initializer='ones', trainable=True)
        self.linear = layers.Dense(units=1, activation=None)
    
    def radial(self, X):
        X = tf.expand_dims(X, axis=1)
        centers = tf.expand_dims(self.centers, axis=0)
        dist = tf.reduce_sum(tf.square(X - centers), axis=-1)
        return tf.exp(-dist / (2 * tf.square(self.widths)))
    
    def call(self, X):
        radial_output = self.radial(X)
        return self.linear(radial_output)

# Define RBF network
model = keras.Sequential([
    RBFLayer(num_centers),
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=2)

# Evaluate model on test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
