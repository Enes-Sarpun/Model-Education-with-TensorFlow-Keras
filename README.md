# 🧠 Deep Learning & Machine Learning 

Welcome to my Deep Learning and Machine Learning repository! This project is a curated collection of various artificial intelligence models, ranging from fundamental neural networks to advanced generative models and reinforcement learning agents. 

The goal of this repository is to demonstrate practical implementations of different deep learning architectures (CNN, RNN, LSTM, GAN, ResNet) and techniques (Transfer Learning, Hyperparameter Tuning, Custom Layers) across diverse datasets.

## 🚀 Technologies & Libraries
* **Python 3.12**
* **TensorFlow / Keras** (Sequential & Functional API, Custom Layers)
* **Scikit-Learn** (Data preprocessing, Evaluation metrics)
* **Gymnasium** (Reinforcement Learning environments)
* **KerasTuner** (Hyperparameter optimization)
* **Pandas, NumPy, Matplotlib, OpenCV** (Data manipulation and visualization)

---

## 📂 Project Index

Here is a breakdown of the scripts included in this repository and the concepts they cover:

### 1. Generative Adversarial Networks (GAN)
* **`mnist.py`**: Implementation of a DCGAN (Deep Convolutional GAN) on the MNIST dataset. It includes a Generator to create synthetic handwritten digits and a Discriminator to distinguish real images from fake ones using `Conv2D` and `Conv2DTranspose` layers.

### 2. Advanced Computer Vision & ResNet
* **`main.py`**: A custom implementation of a **ResNet** (Residual Network) architecture built from scratch. It features custom Residual Blocks and utilizes `KerasTuner` to find the optimal hyperparameters for classifying the Fashion MNIST dataset.

### 3. Transfer Learning
* **`tf.py`**: Image classification pipeline using **MobileNetV2** as a base model for Transfer Learning. Includes extensive data augmentation (ImageDataGenerator), dynamic resizing, and custom classification heads to classify medical/drug vision data.

### 4. Natural Language Processing (NLP)
* **`20news.py`**: Text classification on the 20 Newsgroups dataset using an **LSTM** (Long Short-Term Memory) network. Demonstrates text tokenization, sequence padding, and embedding layers.
* **`imdb.py`**: Sentiment analysis on the IMDB dataset using a **SimpleRNN**. It implements `KerasTuner` for optimizing the embedding output dimensions and RNN units, accompanied by ROC-AUC curve visualizations.

### 5. Reinforcement Learning (RL)
* **`DQL.py`**: A custom **Deep Q-Learning (DQL)** agent built to solve the `CartPole-v1` environment. Features experience replay memory and adaptive epsilon-greedy exploration.
* **`test_agent.py`**: A deployment script to test and evaluate a trained DQL agent on the `LunarLander-v3` environment without retraining, including automated video recording of the agent's performance.

### 6. Image Classification & CNNs
* **`Keras_Cifar_10.py`**: A deep Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. Integrates real-time data augmentation and extensive model evaluation using classification reports.

### 7. Custom Neural Network Layers
* **`iris.py`**: Demonstrates how to write a custom layer in Keras. Implements a custom **Radial Basis Function (RBF)** layer applied to the Iris dataset for tabular data classification.

---

## ⚙️ How to Run

1. **Clone the repository:**
  pip install tensorflow numpy pandas matplotlib scikit-learn gymnasium keras-tuner tqdm imageio
