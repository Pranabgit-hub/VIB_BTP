import matplotlib.pyplot as plt


def plot_accuracy(results):

    methods = list(results.keys())

    values = [
        results[m]["accuracy"]
        for m in methods
    ]

    plt.figure()

    plt.bar(methods, values)

    plt.ylabel("Accuracy")
    plt.xlabel("Divergence")
    plt.title("Classification Accuracy")

    plt.tight_layout()

    plt.savefig(
        "results/plots/accuracy.png"
    )

    plt.close()


def plot_entropy(results, entropy_name):

    methods = list(results.keys())

    values = [
        results[m][entropy_name]
        for m in methods
    ]

    plt.figure()

    plt.bar(methods, values)

    plt.ylabel(entropy_name)
    plt.xlabel("Divergence")
    plt.title(
        f"{entropy_name} of Latent Representation"
    )

    plt.tight_layout()

    plt.savefig(
        f"results/plots/{entropy_name}.png"
    )

    plt.close()


def plot_accuracy_vs_compression(results):

    methods = list(results.keys())

    accuracy = [
        results[m]["accuracy"]
        for m in methods
    ]

    compression = [
        results[m]["information_loss"]
        for m in methods
    ]

    plt.figure()

    for i, method in enumerate(methods):

        plt.scatter(
            compression[i],
            accuracy[i],
            label=method
        )

        plt.annotate(
            method,
            (
                compression[i],
                accuracy[i]
            )
        )

    plt.xlabel("Information / Compression Loss")
    plt.ylabel("Accuracy")
    plt.title(
        "Accuracy vs Information Compression"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/plots/accuracy_vs_compression.png"
    )

    plt.close()