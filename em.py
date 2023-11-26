import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


# Initialization
def initialize_parameters(N, D, K):
    p = np.random.uniform(0, 1, size=(K, D))
    mix_pi = np.full((K, 1), 1/K)
    return p, mix_pi

# E-Step with log for numerical stability
def EStep(X, p, mix_pi, K):
    N = X.shape[0]
    log_eta = np.zeros((N, K))
    
    for i in range(N):
        for k in range(K):
            epsilon = 1e-9  # a small constant
            log_eta[i, k] = np.log(mix_pi[k] + epsilon) + np.sum(X[i] * np.log(p[k] + epsilon) + (1 - X[i]) * np.log(1 - p[k] + epsilon))
            # log_eta[i, k] = np.log(mix_pi[k]) + np.sum(X[i] * np.log(p[k]) + (1 - X[i]) * np.log(1 - p[k]))
            
    # Normalize log_eta for numerical stability
    max_log_eta = np.max(log_eta, axis=1, keepdims=True)
    log_eta = log_eta - max_log_eta
    eta = np.exp(log_eta)
    eta /= np.sum(eta, axis=1, keepdims=True)
    
    return eta

# M-Step
def MStep(X, eta, alpha1, alpha2, K, D):
    N = X.shape[0]
    mix_pi = np.zeros(K)
    p = np.zeros((K, D))
    
    for k in range(K):
        N_k = np.sum(eta[:, k])
        mix_pi[k] = N_k / N
        p[k, :] = (np.dot(eta[:, k], X) + alpha1) / (N_k + alpha1 + alpha2)
        
    return p, mix_pi

# Full EM Algorithm with log for numerical stability
def EM(X, K, iter, alpha1=1e-8, alpha2=1e-8):
    N, D = X.shape
    p, mix_pi = initialize_parameters(N, D, K)
    
    for _ in range(iter):
        eta = EStep(X, p, mix_pi, K)
        p, mix_pi = MStep(X, eta, alpha1, alpha2, K, D)
    
    return p, mix_pi


# Visualization
def visualize_parameters(p):
    
    K = p.shape[0]
    for k in range(K):
        # plt.imshow(p[k].reshape(28, 28), cmap='gray')

        plt.imsave(f"for_{K}/em_{k}.png", p[k].reshape(28, 28), cmap='gray')


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
flatten_images = train_images.reshape(-1, 28*28) / 255.0
binary_images = (flatten_images > 0.5).astype(np.float32)

sample_indices = np.random.choice(binary_images.shape[0], 1000, replace=False)
X = binary_images[sample_indices]
K = 5

p, mix_pi = EM(X, K, iter=20)
visualize_parameters(p)
