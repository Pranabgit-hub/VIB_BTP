DIVERGENCES = [
    "kl",
    "renyi",
    "tsallis"
]
for divergence in DIVERGENCES:

    model = VIB(...)

    train(
        model,
        divergence=divergence
    )

    results = evaluate(model)

    save_results(...)