import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fundbüro Fashion-MNIST", layout="wide")

# ----------------------------
# Labels
# ----------------------------
LABELS = [
    "T-Shirt/Top", "Hose", "Pullover", "Kleid",
    "Mantel", "Sandale", "Hemd", "Sneaker",
    "Tasche", "Stiefel"
]

# ----------------------------
# Daten laden
# ----------------------------
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

# Flatten für Similarity Search
x_train_flat = x_train.reshape(len(x_train), -1)
x_test_flat = x_test.reshape(len(x_test), -1)

# ----------------------------
# Nearest Neighbors Modell
# ----------------------------
@st.cache_resource
def build_nn():
    nn = NearestNeighbors(n_neighbors=6, metric="cosine")
    nn.fit(x_train_flat)
    return nn

nn = build_nn()

# ----------------------------
# UI
# ----------------------------
st.title("🧳 Fundbüro für Kleidung (Fashion-MNIST)")
st.write("Lade ein Bild hoch oder nutze ein Beispiel. Das System findet ähnliche Objekte im Fundbüro.")

option = st.radio("Modus wählen:", ["Zufälliges Beispiel", "Eigenes Bild hochladen"])

def show_results(img, distances, indices):
    cols = st.columns(6)
    for i, idx in enumerate(indices[0]):
        with cols[i]:
            st.image(x_train[idx], caption=LABELS[y_train[idx]], width=100)
            st.write(f"Ähnlichkeit: {1 - distances[0][i]:.2f}")

# ----------------------------
# Zufallsbild
# ----------------------------
if option == "Zufälliges Beispiel":
    idx = np.random.randint(0, len(x_test))
    query = x_test_flat[idx].reshape(1, -1)

    st.subheader("🔎 Gesuchtes Objekt")
    st.image(x_test[idx], width=150, caption=LABELS[y_test[idx]])

    distances, indices = nn.kneighbors(query)
    st.subheader("📦 Gefundene ähnliche Objekte im Fundbüro")
    show_results(x_test[idx], distances, indices)

# ----------------------------
# Upload
# ----------------------------
else:
    uploaded_file = st.file_uploader("Bild hochladen (28x28 Graustufen empfohlen)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = tf.keras.preprocessing.image.load_img(uploaded_file, color_mode="grayscale", target_size=(28, 28))
        img = np.array(img) / 255.0
        query = img.reshape(1, -1)

        st.subheader("🔎 Dein verlorenes Objekt")
        st.image(img, width=150)

        distances, indices = nn.kneighbors(query)

        st.subheader("📦 Ähnliche Fundstücke")
        show_results(img, distances, indices)

# ----------------------------
# Info
# ----------------------------
st.sidebar.title("ℹ️ Info")
st.sidebar.write("""
Dieses Fundbüro nutzt Fashion-MNIST:
- 60.000 Trainingsbilder
- 10 Kleidungs-Kategorien
- Ähnlichkeit via k-NN (Cosine Distance)
""")
