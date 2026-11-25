"""Train and compare two 2D flow maps on the same two-moons samples."""

import argparse
import math

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


def two_moons(count, rng):
    angle = math.pi * torch.rand(count, generator=rng)
    outer = torch.stack((angle.cos(), angle.sin()), dim=1)
    inner = torch.stack((1 - angle.cos(), 0.5 - angle.sin()), dim=1)
    selector = torch.rand(count, 1, generator=rng) < 0.5
    noise = 0.06 * torch.randn(count, 2, generator=rng)
    return torch.where(selector, outer, inner) + noise


def features(x, t):
    return torch.cat((x, t, (2*t).sin(), (2*t).cos(), (4*t).sin(), (4*t).cos()), dim=1)


def train(data, coupling, steps, seed):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(7, 96), nn.SiLU(),
        nn.Linear(96, 96), nn.SiLU(),
        nn.Linear(96, 96), nn.SiLU(),
        nn.Linear(96, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        target = data[torch.randint(len(data), (128,), generator=rng)]
        prior = torch.randn(128, 2, generator=rng)
        t = torch.rand(128, 1, generator=rng)
        if coupling == "minibatch-ot":
            row, col = linear_sum_assignment(torch.cdist(prior, target).square().numpy())
            prior, target = prior[row], target[col]
        interpolated = (1 - t) * prior + t * target
        loss = nn.functional.mse_loss(model(features(interpolated, t)), target - prior)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.no_grad()
def generate(model, prior, steps):
    x = prior.clone()
    for step in range(steps):
        t = torch.full((x.shape[0], 1), step / steps)
        x += model(features(x, t)) / steps
    return x


def sliced_w2(samples, target, directions):
    projected = (samples @ directions).sort(dim=0).values
    reference = (target @ directions).sort(dim=0).values
    return (projected - reference).square().mean().sqrt().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    torch.set_num_threads(2)
    data = two_moons(12_000, torch.Generator().manual_seed(404))
    rng = torch.Generator().manual_seed(405)
    test = two_moons(2_000, rng)
    prior = torch.randn(2_000, 2, generator=rng)
    directions = nn.functional.normalize(torch.randn(2, 128, generator=rng), dim=0)
    print("coupling,euler_steps,sliced_w2")
    for coupling in ("random", "minibatch-ot"):
        model = train(data, coupling, args.steps, args.seed)
        for steps in (5, 10, 20, 50, 100):
            value = sliced_w2(generate(model, prior, steps), test, directions)
            print(f"{coupling},{steps},{value:.6f}")


if __name__ == "__main__":
    main()
