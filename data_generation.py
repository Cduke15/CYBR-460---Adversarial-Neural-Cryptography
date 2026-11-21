import numpy as np

def create_dataset(num_samples, bit_len=16, seed=42):
    rng = np.random.default_rng(seed)
    messages = rng.choice([-1.0, 1.0], size=(num_samples, bit_len)).astype(np.float32)
    keys = rng.choice([-1.0, 1.0], size=(num_samples, bit_len)).astype(np.float32)
    return messages, keys

if __name__ == "__main__":
    train_m, train_k = create_dataset(10000, seed=42)
    test_m, test_k = create_dataset(1000, seed=99)
    np.save("train_messages.npy", train_m)
    np.save("train_keys.npy", train_k)
    np.save("test_messages.npy", test_m)
    np.save("test_keys.npy", test_k)
    print("Datasets saved successfully.")
