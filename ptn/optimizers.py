"""
Second-order optimization utilities for tensor networks.

This module provides utilities for creating and using second-order optimization
methods with tensor network models, particularly MPS (Matrix Product State) models.
"""

import torch
from typing import Optional, Union, Dict, Any


def create_optimizer(
    model: torch.nn.Module, optimizer_name: str, lr: float = 1e-3, **kwargs
) -> torch.optim.Optimizer:
    """
    Create an optimizer with appropriate parameters for tensor network training.

    Args:
        model: The model to optimize
        optimizer_name: Name of the optimizer ('AdamW', 'SGD', 'LBFGS', 'AdaHessian')
        lr: Learning rate
        **kwargs: Additional optimizer-specific parameters

    Returns:
        Configured optimizer

    Raises:
        ImportError: If AdaHessian is requested but not available
        ValueError: If optimizer_name is not supported
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-8),
            weight_decay=kwargs.get("weight_decay", 0.01),
        )

    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=kwargs.get("momentum", 0.9),
            weight_decay=kwargs.get("weight_decay", 0.0),
        )

    elif optimizer_name == "lbfgs":
        return torch.optim.LBFGS(
            model.parameters(),
            lr=kwargs.get("lr", 1.0),  # L-BFGS typically uses lr=1.0
            max_iter=kwargs.get("max_iter", 20),
            max_eval=kwargs.get("max_eval", 25),
            history_size=kwargs.get("history_size", 100),
            line_search_fn=kwargs.get("line_search_fn", "strong_wolfe"),
        )

    elif optimizer_name == "adahessian":
        try:
            import adahessian
        except ImportError:
            raise ImportError(
                "AdaHessian not available. Install with: pip install adahessian"
            )

        return adahessian.AdaHessian(
            model.parameters(),
            lr=lr,
            betas=kwargs.get("betas", (0.9, 0.999)),
            eps=kwargs.get("eps", 1e-4),
            weight_decay=kwargs.get("weight_decay", 0.0),
            hessian_power=kwargs.get("hessian_power", 0.5),
            update_each=kwargs.get("update_each", 1),
            n_samples=kwargs.get("n_samples", 1),
            average_conv_kernel=kwargs.get("average_conv_kernel", False),
        )

    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. "
            f"Supported options: 'adamw', 'sgd', 'lbfgs', 'adahessian'"
        )


def get_optimizer_recommendations(
    dataset_size: int,
    model_complexity: str = "medium",
    memory_limited: bool = False,
    convergence_speed_priority: bool = False,
) -> Dict[str, Any]:
    """
    Get optimizer recommendations based on problem characteristics.

    Args:
        dataset_size: Number of training samples
        model_complexity: 'low', 'medium', or 'high'
        memory_limited: Whether memory is a constraint
        convergence_speed_priority: Whether fast convergence is more important than stability

    Returns:
        Dictionary with optimizer recommendations and parameters
    """
    recommendations = {}

    # Small datasets: L-BFGS is often best
    if dataset_size < 1000:
        recommendations["primary"] = {
            "name": "LBFGS",
            "lr": 1.0,
            "max_iter": 20,
            "max_eval": 25,
            "history_size": 100,
            "reason": "Fast convergence on small datasets with smooth landscapes",
        }
        recommendations["fallback"] = {
            "name": "AdaHessian",
            "lr": 1e-3,
            "hessian_power": 0.5,
            "reason": "Robust second-order method with adaptive learning",
        }

    # Medium datasets: AdaHessian or AdamW
    elif dataset_size < 10000:
        if convergence_speed_priority and not memory_limited:
            recommendations["primary"] = {
                "name": "AdaHessian",
                "lr": 1e-3,
                "hessian_power": 0.5,
                "reason": "Good balance of speed and robustness",
            }
        else:
            recommendations["primary"] = {
                "name": "AdamW",
                "lr": 1e-3,
                "weight_decay": 0.01,
                "reason": "Robust and well-tested for medium datasets",
            }
        recommendations["fallback"] = {
            "name": "LBFGS",
            "lr": 1.0,
            "max_iter": 15,
            "reason": "Fast convergence if memory allows",
        }

    # Large datasets: AdamW or SGD
    else:
        recommendations["primary"] = {
            "name": "AdamW",
            "lr": 1e-3,
            "weight_decay": 0.01,
            "reason": "Most robust for large datasets",
        }
        if convergence_speed_priority:
            recommendations["fallback"] = {
                "name": "SGD",
                "lr": 1e-2,
                "momentum": 0.9,
                "reason": "Fast convergence with proper tuning",
            }

    # Memory-limited overrides
    if memory_limited:
        recommendations["primary"] = {
            "name": "AdamW",
            "lr": 1e-3,
            "weight_decay": 0.01,
            "reason": "Memory efficient",
        }
        recommendations["fallback"] = {
            "name": "SGD",
            "lr": 1e-2,
            "momentum": 0.9,
            "reason": "Lowest memory usage",
        }

    return recommendations


def create_lbfgs_closure(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: callable,
) -> callable:
    """
    Create a closure function for L-BFGS optimization.

    Args:
        model: The model to optimize
        dataloader: Data loader for training data
        device: Device to run computation on
        loss_fn: Loss function that takes (model, batch) and returns loss

    Returns:
        Closure function for L-BFGS
    """

    def closure():
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            loss = loss_fn(model, batch, device)
            loss.backward()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    return closure


def get_optimizer_info(optimizer_name: str) -> Dict[str, Any]:
    """
    Get detailed information about an optimizer.

    Args:
        optimizer_name: Name of the optimizer

    Returns:
        Dictionary with optimizer information
    """
    optimizer_name = optimizer_name.lower()

    info = {
        "adamw": {
            "type": "First-order",
            "description": "Adam with decoupled weight decay",
            "pros": [
                "Robust",
                "Good for large datasets",
                "Handles noisy gradients well",
            ],
            "cons": ["Slower convergence", "Hyperparameter sensitive"],
            "best_for": ["Large datasets", "Noisy data", "General purpose"],
            "memory_usage": "Medium",
            "convergence_speed": "Medium",
        },
        "sgd": {
            "type": "First-order",
            "description": "Stochastic Gradient Descent with momentum",
            "pros": ["Simple", "Memory efficient", "Fast with good tuning"],
            "cons": ["Requires careful tuning", "Sensitive to learning rate"],
            "best_for": [
                "Large datasets",
                "Memory constrained",
                "Well-tuned scenarios",
            ],
            "memory_usage": "Low",
            "convergence_speed": "High (with tuning)",
        },
        "lbfgs": {
            "type": "Second-order (Quasi-Newton)",
            "description": "Limited-memory Broyden-Fletcher-Goldfarb-Shanno",
            "pros": [
                "Fast convergence",
                "No hyperparameter tuning",
                "Memory efficient for second-order",
            ],
            "cons": [
                "Requires full gradients",
                "Sensitive to line search",
                "Not suitable for large datasets",
            ],
            "best_for": ["Small datasets", "Smooth landscapes", "Fast prototyping"],
            "memory_usage": "Medium-High",
            "convergence_speed": "Very High",
        },
        "adahessian": {
            "type": "Second-order (Adaptive)",
            "description": "Adaptive Hessian-based optimizer",
            "pros": [
                "Combines Adam robustness with second-order benefits",
                "Adaptive learning",
            ],
            "cons": ["Higher computational cost", "More memory usage"],
            "best_for": ["Medium datasets", "When second-order benefits are desired"],
            "memory_usage": "High",
            "convergence_speed": "High",
        },
    }

    return info.get(
        optimizer_name,
        {
            "type": "Unknown",
            "description": "Unknown optimizer",
            "pros": [],
            "cons": [],
            "best_for": [],
            "memory_usage": "Unknown",
            "convergence_speed": "Unknown",
        },
    )
