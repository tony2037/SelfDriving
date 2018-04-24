from loader import loader

l = loader.loader(x_path="./dataset/preprocess_image/x", y_path="./dataset/preprocess_image/ys.npy")

data = l.next_batch(batch_size=8)
