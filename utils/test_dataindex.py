import numpy as np
img_vecs = np.load("D:/EEG/egg_encoder/data/image_vectors.npy")
txt_vecs = np.load("D:/EEG/egg_encoder/data/text_vectors.npy")
print("Image vectors shape:", img_vecs.shape)
print("Text vectors shape:", txt_vecs.shape)