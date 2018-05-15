from loader import loader

l = loader.loader(x_path="./dataset/preprocess_image/x", y_path="./dataset/preprocess_image/ys.npy")

data = l.next_batch(batch_size=1)

#print(data)

for i in range(5):
    t = l.next_batch(batch_size=1)
    print(t)