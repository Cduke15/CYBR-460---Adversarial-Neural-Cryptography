import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def to_binary_str(bipolar_vec):
    """Convert {-1, 1} float array into a readable '0101...' string."""
    bits = (bipolar_vec > 0).astype(int)
    return "".join(str(b) for b in bits)

def compute_bit_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def run_evaluation_and_demo():
    # Loads model for tests
    alice = tf.keras.models.load_model("alice.keras")
    bob = tf.keras.models.load_model("bob.keras")
    eve = tf.keras.models.load_model("eve.keras")

    test_m = np.load("test_messages.npy")
    test_k = np.load("test_keys.npy")

    # Evaluation of 1,000 unseen samples
    ak_in = np.concatenate([test_m, test_k], axis=1)
    ciphertext = alice.predict(ak_in, verbose=0)
    bob_out = bob.predict(np.concatenate([ciphertext, test_k], axis=1), verbose=0)
    eve_out = eve.predict(ciphertext, verbose=0)

    bob_acc = compute_bit_accuracy(test_m, bob_out) * 100
    eve_acc = compute_bit_accuracy(test_m, eve_out) * 100

    print("=" * 55)
    print("FINAL EVALUATION (1,000 Unseen Test Samples)")
    print(f"Bob Bit Accuracy : {bob_acc:.2f}%  (Target: >95.00%)")
    print(f"Eve Bit Accuracy : {eve_acc:.2f}%  (Target: ~50.00%)")
    print("=" * 55)

    # Demo printouts
    print("\nEvaluation demo")
    for i in range(3):
        orig_bin = to_binary_str(test_m[i])
        key_bin = to_binary_str(test_k[i])
        bob_bin = to_binary_str(bob_out[i])
        eve_bin = to_binary_str(eve_out[i])
        cipher_preview = np.round(ciphertext[i][:6], 2)

        print(f"\nSample #{i + 1}")
        print(f"  Original Plaintext : {orig_bin}")
        print(f"  Secret Key         : {key_bin}")
        print(f"  Alice Ciphertext   : {cipher_preview} ... (16 continuous floats)")
        print(f"  Bob Decrypted      : {bob_bin}  ({'MATCH' if orig_bin == bob_bin else 'DIFF'})")
        print(f"  Eve Guessed        : {eve_bin}  ({'CRACKED' if orig_bin == eve_bin else 'FAILED'})")

def plot_training_curves():
    # Generates visualizations
    history = np.load("history.npy", allow_pickle=True).item()
    epochs = range(1, len(history["bob_acc"]) + 1)

    # Plot 1: Bit Accuracy Over Time (for png)
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs, [a * 100 for a in history["bob_acc"]], label="Bob (Authorized)", linewidth=2.5, color="#1f77b4")
    plt.plot(epochs, [a * 100 for a in history["eve_acc"]], label="Eve (Eavesdropper)", linewidth=2.5, color="#d62728")
    plt.axhline(50, color="gray", linestyle="--", linewidth=1, label="Random Guess Baseline (50%)")
    plt.title("Adversarial Training: Bit Accuracy Over Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Bit Accuracy (%)", fontsize=11)
    plt.ylim(35, 102)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("accuracy_curve.png")
    plt.close()

    # Plot 2: L1 Loss Over Time (for png)
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(epochs, history["bob_loss"], label="Bob Reconstruction Loss", linewidth=2.5, color="#1f77b4")
    plt.plot(epochs, history["eve_loss"], label="Eve Reconstruction Loss", linewidth=2.5, color="#d62728")
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Ideal Eve L1 Error (1.0)")
    plt.title("Adversarial Training: L1 Loss Over Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("L1 Loss", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()
    print("\nSaved 'accuracy_curve.png' and 'loss_curve.png'.")

if __name__ == "__main__":
    run_evaluation_and_demo()
    plot_training_curves()
