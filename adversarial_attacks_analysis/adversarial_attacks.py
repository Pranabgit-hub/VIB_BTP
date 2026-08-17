import torch
import torch.nn.functional as F


# =========================================================
# FGSM ATTACK
# =========================================================

def fgsm_attack(
    model,
    images,
    labels,
    epsilon,
    device
):
    """
    Fast Gradient Sign Method (FGSM).

    Generates adversarial examples using a single gradient
    step:

        x' = x + epsilon * sign(grad_x L(f(x), y))

    The attack uses the DETERMINISTIC encoder output
    (mean only, no reparameterization noise) so that the
    gradient signal is clean and the robustness measurement
    is not confused by stochastic sampling.

    Args:
        model:    trained VIB model
        images:   input images, shape (B, 784)
        labels:   true labels, shape (B,)
        epsilon:  perturbation magnitude
        device:   torch device

    Returns:
        adversarial_images: perturbed inputs, shape (B, 784)
    """

    images = images.clone().detach().to(device)

    labels = labels.clone().detach().to(device)

    images.requires_grad = True


    # -------------------------------------------------
    # Deterministic forward pass (no sampling noise)
    #
    # Use encoder mean directly through the classifier.
    # -------------------------------------------------

    mu, logvar = model.encode(images)

    logits = model.classifier(mu)

    loss = F.cross_entropy(
        logits,
        labels
    )


    # -------------------------------------------------
    # Compute gradient w.r.t. input
    # -------------------------------------------------

    model.zero_grad()

    loss.backward()

    data_grad = images.grad.data


    # -------------------------------------------------
    # Create adversarial example
    # -------------------------------------------------

    sign_data_grad = data_grad.sign()

    adversarial_images = (
        images.data
        + epsilon * sign_data_grad
    )

    # Clamp to valid pixel range [0, 1]
    adversarial_images = torch.clamp(
        adversarial_images,
        0.0,
        1.0
    )

    return adversarial_images.detach()


# =========================================================
# PGD ATTACK
# =========================================================

def pgd_attack(
    model,
    images,
    labels,
    epsilon,
    device,
    num_steps=20,
    step_size=None
):
    """
    Projected Gradient Descent (PGD) attack.

    Multi-step iterative attack with random start:

        x_0 = x + uniform(-epsilon, epsilon)

        x_{t+1} = project(
            x_t + step_size * sign(grad L),
            B(x, epsilon)
        )

    Uses deterministic encoder output (mean only) for
    clean gradient signal.

    Args:
        model:      trained VIB model
        images:     input images, shape (B, 784)
        labels:     true labels, shape (B,)
        epsilon:    maximum perturbation magnitude
        device:     torch device
        num_steps:  number of PGD iterations
        step_size:  per-step size (default: epsilon / 4)

    Returns:
        adversarial_images: perturbed inputs, shape (B, 784)
    """

    if step_size is None:
        step_size = epsilon / 4.0


    original_images = (
        images.clone().detach().to(device)
    )

    labels = labels.clone().detach().to(device)


    # -------------------------------------------------
    # Random start within epsilon ball
    # -------------------------------------------------

    adversarial_images = (
        original_images
        + torch.empty_like(original_images).uniform_(
            -epsilon,
            epsilon
        )
    )

    adversarial_images = torch.clamp(
        adversarial_images,
        0.0,
        1.0
    )


    # -------------------------------------------------
    # Iterative PGD steps
    # -------------------------------------------------

    for _ in range(num_steps):

        adversarial_images = (
            adversarial_images.clone().detach()
        )

        adversarial_images.requires_grad = True


        # Deterministic forward (mean only)
        mu, logvar = model.encode(
            adversarial_images
        )

        logits = model.classifier(mu)

        loss = F.cross_entropy(
            logits,
            labels
        )


        model.zero_grad()

        loss.backward()

        data_grad = adversarial_images.grad.data


        # Gradient ascent step
        adversarial_images = (
            adversarial_images.data
            + step_size * data_grad.sign()
        )


        # Project back into epsilon ball
        perturbation = (
            adversarial_images
            - original_images
        )

        perturbation = torch.clamp(
            perturbation,
            -epsilon,
            epsilon
        )

        adversarial_images = (
            original_images + perturbation
        )


        # Clamp to valid pixel range
        adversarial_images = torch.clamp(
            adversarial_images,
            0.0,
            1.0
        )


    return adversarial_images.detach()


# =========================================================
# EVALUATE UNDER ATTACK
# =========================================================

def evaluate_under_attack(
    model,
    attack_fn,
    test_loader,
    epsilon,
    device
):
    """
    Evaluate a model's accuracy under a given attack.

    Uses deterministic forward pass (encoder mean, no
    sampling) for both attack generation and accuracy
    evaluation.

    Args:
        model:        trained VIB model (in eval mode)
        attack_fn:    callable(model, images, labels, epsilon, device)
        test_loader:  DataLoader for test set
        epsilon:      perturbation magnitude
        device:       torch device

    Returns:
        accuracy: float, adversarial accuracy
    """

    model.eval()

    correct = 0
    total = 0


    for images, labels in test_loader:

        images = images.view(
            images.size(0),
            -1
        ).to(device)

        labels = labels.to(device)


        # Generate adversarial examples
        adversarial_images = attack_fn(
            model,
            images,
            labels,
            epsilon,
            device
        )


        # Evaluate on adversarial examples
        # (deterministic: use mean, no sampling)
        with torch.no_grad():

            mu, logvar = model.encode(
                adversarial_images
            )

            logits = model.classifier(mu)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


    accuracy = correct / total

    return accuracy
