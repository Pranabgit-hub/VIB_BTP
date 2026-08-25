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
    device,
    clamp_output=True
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
        model:        trained VIB model
        images:       inputs in model input format — flat (B, D) or image (B, C, H, W)
        labels:       true labels, shape (B,)
        epsilon:      perturbation magnitude
        device:       torch device
        clamp_output: if True clamp to [0, 1] (images);
                      if False leave unperturbed (time series)

    Returns:
        adversarial_images: perturbed inputs, same shape as images
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

    # Clamp to valid pixel range [0, 1] only for images
    if clamp_output:
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
    step_size=None,
    clamp_output=True
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
        model:        trained VIB model
        images:       inputs in model input format — flat (B, D) or image (B, C, H, W)
        labels:       true labels, shape (B,)
        epsilon:      maximum perturbation magnitude
        device:       torch device
        num_steps:    number of PGD iterations
        step_size:    per-step size (default: epsilon / 4)
        clamp_output: if True clamp to [0, 1] (images);
                      if False leave unperturbed (time series)

    Returns:
        adversarial_images: perturbed inputs, same shape as images
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

    if clamp_output:
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
        if clamp_output:
            adversarial_images = torch.clamp(
                adversarial_images,
                0.0,
                1.0
            )


    return adversarial_images.detach()


# =========================================================
# TRAINING-TIME ADVERSARIAL EXAMPLE GENERATION
#
# Used for ADVERSARIAL TRAINING (not evaluation).
# Same inner-maximization logic as the eval attacks above,
# with three differences that matter during training:
#
#   1. torch.autograd.grad(loss, images) is used instead of
#      loss.backward(), so parameter .grad buffers are never
#      touched — the optimizer step that follows only sees
#      gradients from the actual training loss.
#
#   2. The model is temporarily switched to eval mode while
#      crafting examples, so BatchNorm running statistics are
#      not updated by these extra forward passes, then the
#      previous mode is restored.
#
#   3. Supports both "fgsm" (single-step) and "pgd"
#      (multi-step Madry-style) inner maximization, so the
#      same entry point drives FGSM- and PGD-adversarial
#      training.
# =========================================================

def generate_adversarial_batch(
    model,
    images,
    labels,
    epsilon,
    device,
    method="fgsm",
    num_steps=5,
    step_size=None,
    clamp_output=True
):

    if method not in ("fgsm", "pgd"):

        raise ValueError(
            f"Unknown adversarial-training method: {method}. "
            f"Choose from: ['fgsm', 'pgd']"
        )

    was_training = model.training

    model.eval()


    try:

        images = images.clone().detach().to(device)

        labels = labels.clone().detach().to(device)

        if method == "fgsm":

            adversarial_images = _fgsm_inner_max(
                model,
                images,
                labels,
                epsilon,
                clamp_output
            )

        else:

            adversarial_images = _pgd_inner_max(
                model,
                images,
                labels,
                epsilon,
                num_steps=num_steps,
                step_size=step_size,
                clamp_output=clamp_output
            )


    finally:

        if was_training:

            model.train()


    return adversarial_images.detach()


def _fgsm_inner_max(
    model,
    images,
    labels,
    epsilon,
    clamp_output=True
):
    """
    One-step inner maximization (FGSM).
    """

    perturbed = images.clone().detach()

    perturbed.requires_grad_(True)


    mu, logvar = model.encode(perturbed)

    logits = model.classifier(mu)

    loss = F.cross_entropy(logits, labels)


    data_grad = torch.autograd.grad(
        loss,
        perturbed
    )[0]


    adversarial_images = (
        images + epsilon * data_grad.sign()
    )

    if clamp_output:
        return torch.clamp(
            adversarial_images,
            0.0,
            1.0
        )

    return adversarial_images


def _pgd_inner_max(
    model,
    images,
    labels,
    epsilon,
    num_steps=5,
    step_size=None,
    clamp_output=True
):
    """
    Multi-step inner maximization (PGD, random start).

    Default step size epsilon / 4 — for CIFAR-10 at
    eps = 8/255 this equals 2/255, the standard Madry
    recipe.
    """

    if step_size is None:

        step_size = epsilon / 4.0


    # Random start inside the epsilon ball
    adversarial_images = (
        images
        + torch.empty_like(images).uniform_(
            -epsilon,
            epsilon
        )
    )

    if clamp_output:
        adversarial_images = torch.clamp(
            adversarial_images,
            0.0,
            1.0
        )


    for _ in range(num_steps):

        perturbed = adversarial_images.clone().detach()

        perturbed.requires_grad_(True)


        mu, logvar = model.encode(perturbed)

        logits = model.classifier(mu)

        loss = F.cross_entropy(logits, labels)


        data_grad = torch.autograd.grad(
            loss,
            perturbed
        )[0]


        # Gradient ascent step
        adversarial_images = (
            adversarial_images.detach()
            + step_size * data_grad.sign()
        )


        # Project onto the epsilon ball around x
        perturbation = (
            adversarial_images - images
        )

        perturbation = torch.clamp(
            perturbation,
            -epsilon,
            epsilon
        )

        adversarial_images = (
            images + perturbation
        )


        adversarial_images = torch.clamp(
            adversarial_images,
            0.0,
            1.0
        )


    return adversarial_images


# =========================================================
# EVALUATE UNDER ATTACK
# =========================================================

def evaluate_under_attack(
    model,
    attack_fn,
    test_loader,
    epsilon,
    device,
    clamp_output=True
):
    """
    Evaluate a model's accuracy under a given attack.

    Uses deterministic forward pass (encoder mean, no
    sampling) for both attack generation and accuracy
    evaluation.

    Batches from test_loader are used exactly as yielded
    — the loader's collate_fn is responsible for putting
    them in the model's expected input format (flattened
    vectors for the MLP, image tensors for the CNN).

    Args:
        model:        trained VIB model (in eval mode)
        attack_fn:    callable(model, images, labels, epsilon, device)
        test_loader:  DataLoader for test set
        epsilon:      perturbation magnitude
        device:       torch device
        clamp_output: passed through to attack_fn

    Returns:
        accuracy: float, adversarial accuracy
    """

    model.eval()

    correct = 0
    total = 0


    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        # Generate adversarial examples
        adversarial_images = attack_fn(
            model,
            images,
            labels,
            epsilon,
            device,
            clamp_output=clamp_output
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