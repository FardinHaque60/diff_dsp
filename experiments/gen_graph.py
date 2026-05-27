import matplotlib.pyplot as plt

def gen_plots(weights, loss_history):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.title("Loss over Time")
    plt.plot(loss_history, color='red')

    plt.subplot(2, 1, 2)
    plt.title("Learned Filter Weights (The FIR Kernel)")
    plt.plot(weights, marker='o')

    plt.tight_layout()
    plt.show()