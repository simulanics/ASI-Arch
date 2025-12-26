#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test trainer for GateFlooredResNet attention mechanism (DeltaNet layer)
Demonstrates a simple training loop with MSE loss on random data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from GateFlooredResNet import DeltaNet


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 4
    seq_len = 16
    hidden_size = 128
    num_heads = 4
    num_epochs = 10
    learning_rate = 1e-3

    # Initialize model
    model = DeltaNet(
        mode="cagf_br",
        d_model=hidden_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        use_beta=True,
        use_gate=False,
        use_short_conv=True,
        conv_size=3,
        conv_bias=False,
        fir_kernel_size_long=8,
        fir_kernel_size_short=3,
        prob_floor=0.02
    ).to(device)

    # Create dummy inputs and targets (autoencoder-style)
    inputs = torch.randn(batch_size, seq_len, hidden_size, device=device)
    targets = inputs.clone()

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        outputs, _, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
