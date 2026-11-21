import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

BIT_LEN = 16
BATCH_SIZE = 64
EPOCHS = 60
LR = 0.001

def build_symmetric_net(name):
    # Increased the capacity so Alice/Bob outpace Eve, here
    return models.Sequential([
        layers.Input(shape=(BIT_LEN * 2,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(BIT_LEN, activation="tanh")
    ], name=name)

def build_eve_net():
    # Eve was too good - added GaussianNoise to distort, here
    return models.Sequential([
        layers.Input(shape=(BIT_LEN,)),
        layers.GaussianNoise(0.05),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(BIT_LEN, activation="tanh")
    ], name="Eve")

alice = build_symmetric_net("Alice")
bob = build_symmetric_net("Bob")
eve = build_eve_net()

opt_ab = tf.keras.optimizers.Adam(LR)
opt_eve = tf.keras.optimizers.Adam(LR)

def bit_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_true), tf.sign(y_pred)), tf.float32))

@tf.function
def train_step(messages, keys):
    # Training Eve
    for _ in range(2):
        with tf.GradientTape() as tape_eve:
            ak_in = tf.concat([messages, keys], axis=1)
            ciphertext = alice(ak_in, training=True)
            eve_guess = eve(ciphertext, training=True)
            loss_eve = tf.reduce_mean(tf.abs(messages - eve_guess))
        grads_eve = tape_eve.gradient(loss_eve, eve.trainable_variables)
        opt_eve.apply_gradients(zip(grads_eve, eve.trainable_variables))

    # Freezes Eve & Train Alice and Bob
    with tf.GradientTape() as tape_ab:
        ak_in = tf.concat([messages, keys], axis=1)
        ciphertext = alice(ak_in, training=True)
        bob_guess = bob(tf.concat([ciphertext, keys], axis=1), training=True)
        eve_guess = eve(ciphertext, training=False)

        loss_bob = tf.reduce_mean(tf.abs(messages - bob_guess))
        loss_eve_eval = tf.reduce_mean(tf.abs(messages - eve_guess))
        # Stronger penalty keeps Eve pinned near random guessing (L1 = 1.0)
        loss_ab = loss_bob + 2.5 * tf.square(1.0 - loss_eve_eval)

    ab_vars = alice.trainable_variables + bob.trainable_variables
    grads_ab = tape_ab.gradient(loss_ab, ab_vars)
    opt_ab.apply_gradients(zip(grads_ab, ab_vars))

    return loss_bob, loss_eve_eval, bit_accuracy(messages, bob_guess), bit_accuracy(messages, eve_guess)

if __name__ == "__main__":
    train_m = np.load("train_messages.npy")
    train_k = np.load("train_keys.npy")
    val_m = np.load("test_messages.npy")
    val_k = np.load("test_keys.npy")

    dataset = tf.data.Dataset.from_tensor_slices((train_m, train_k)).shuffle(10000).batch(BATCH_SIZE)
    history = {"bob_acc": [], "eve_acc": [], "bob_loss": [], "eve_loss": []}

    for epoch in range(1, EPOCHS + 1):
        b_accs, e_accs, b_losses, e_losses = [], [], [], []
        for m_batch, k_batch in dataset:
            lb, le, ba, ea = train_step(m_batch, k_batch)
            b_losses.append(lb.numpy())
            e_losses.append(le.numpy())
            b_accs.append(ba.numpy())
            e_accs.append(ea.numpy())

        history["bob_acc"].append(np.mean(b_accs))
        history["eve_acc"].append(np.mean(e_accs))
        history["bob_loss"].append(np.mean(b_losses))
        history["eve_loss"].append(np.mean(e_losses))

        # 5 jumps - keeps the data between but also helps keep things clean
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Bob Acc: {history['bob_acc'][-1]:.4f} | Eve Acc: {history['eve_acc'][-1]:.4f}")

    # Validation check
    val_cipher = alice(tf.concat([val_m, val_k], axis=1), training=False)
    val_bob = bob(tf.concat([val_cipher, val_k], axis=1), training=False)
    val_eve = eve(val_cipher, training=False)
    print(f"\nValidation Set -> Bob Acc: {bit_accuracy(val_m, val_bob):.4f} | Eve Acc: {bit_accuracy(val_m, val_eve):.4f}")

    alice.save("alice.keras")
    bob.save("bob.keras")
    eve.save("eve.keras")
    np.save("history.npy", history)
