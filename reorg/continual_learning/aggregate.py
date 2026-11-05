# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tie-aware aggregation of groupwise rankings (App. A.2, "Tie-aware aggregation").

Every judged group is expanded into pairs: different tiers give a win and a
loss, a shared tier gives a tie. For centered candidate log strengths theta_i
and a log tie parameter eta, a Davidson model

    (p_{i>j}, p_{j>i}, p_{i~j}) = softmax(theta_i, theta_j, eta + (theta_i + theta_j) / 2)

is fitted to the mean negative log likelihood plus
0.05 sum_i theta_i^2 + 0.05 eta^2, with sum_i theta_i = 0, by deterministic
Adam (learning rate 0.05, coefficients (0.9, 0.999), stabilizer 1e-8, at most
4000 iterations). The fit stops at a maximum gradient of 1e-8, or once the
loss changes by at most 1e-12 while the maximum gradient is at most 1e-6.
Normalized strengths exp(theta_i) / sum_j exp(theta_j) give the task-local
ranking; a candidate within 1e-6 of its tier's first member shares that tier.
The fit uses plain Python floats in a fixed operation order, so it is
deterministic up to the platform's exp/log.
"""

from __future__ import annotations

import itertools
import math

REG = 0.05
LEARNING_RATE = 0.05
BETAS = (0.9, 0.999)
EPSILON = 1e-8
MAX_ITERATIONS = 4000
GRADIENT_TOLERANCE = 1e-8
LOSS_TOLERANCE = 1e-12
STALL_GRADIENT = 1e-6
TIER_TOLERANCE = 1e-6


def pairs(groups: list[dict]) -> list[tuple[str, str, int]]:
    """(left, right, outcome) for each pair of members, in membership order; outcome 0 means left
    wins, 1 right wins, 2 a tie. A group is {"candidates": [...], "tiers": [[...], ...]}."""
    rows = []
    for group in groups:
        tier = {candidate: index for index, members in enumerate(group["tiers"]) for candidate in members}
        for left, right in itertools.combinations(group["candidates"], 2):
            rows.append((left, right, 0 if tier[left] < tier[right] else 1 if tier[right] < tier[left] else 2))
    return rows


def _softmax(logits: tuple[float, float, float]) -> tuple[float, ...]:
    top = max(logits)
    values = [math.exp(value - top) for value in logits]
    total = sum(values)
    return tuple(value / total for value in values)


def fit(candidates: list[str], observations: list[tuple[str, str, int]], *, reg: float = REG,
        learning_rate: float = LEARNING_RATE, max_iterations: int = MAX_ITERATIONS) -> tuple[dict, float, dict]:
    """(theta by candidate, eta, diagnostics)."""
    index = {candidate: position for position, candidate in enumerate(candidates)}
    size, count = len(candidates), float(len(observations))
    theta, eta = [0.0] * size, 0.0
    first, second, first_eta, second_eta = [0.0] * size, [0.0] * size, 0.0, 0.0
    beta1, beta2 = BETAS
    converged, last_loss, max_gradient, iteration = False, math.inf, math.inf, 0
    for iteration in range(1, max_iterations + 1):
        gradient, gradient_eta, loss = [0.0] * size, 0.0, 0.0
        for left, right, outcome in observations:
            i, j = index[left], index[right]
            probabilities = _softmax((theta[i], theta[j], eta + 0.5 * (theta[i] + theta[j])))
            loss -= math.log(max(probabilities[outcome], 1e-300))
            residual = [probabilities[k] - int(k == outcome) for k in range(3)]
            gradient[i] += residual[0] + 0.5 * residual[2]
            gradient[j] += residual[1] + 0.5 * residual[2]
            gradient_eta += residual[2]
        loss /= count
        gradient = [value / count for value in gradient]
        gradient_eta /= count
        for position, value in enumerate(theta):
            loss += reg * value * value
            gradient[position] += 2 * reg * value
        loss += reg * eta * eta
        gradient_eta += 2 * reg * eta
        mean = sum(gradient) / size                      # project onto sum(theta) = 0
        gradient = [value - mean for value in gradient]
        max_gradient = max([abs(value) for value in gradient] + [abs(gradient_eta)])
        if max_gradient <= GRADIENT_TOLERANCE:
            converged, last_loss = True, loss
            break
        for position, value in enumerate(gradient):
            first[position] = beta1 * first[position] + (1.0 - beta1) * value
            second[position] = beta2 * second[position] + (1.0 - beta2) * value * value
            corrected = first[position] / (1.0 - beta1 ** iteration)
            theta[position] -= learning_rate * corrected / (math.sqrt(second[position] / (1.0 - beta2 ** iteration))
                                                            + EPSILON)
        first_eta = beta1 * first_eta + (1.0 - beta1) * gradient_eta
        second_eta = beta2 * second_eta + (1.0 - beta2) * gradient_eta * gradient_eta
        corrected = first_eta / (1.0 - beta1 ** iteration)
        eta -= learning_rate * corrected / (math.sqrt(second_eta / (1.0 - beta2 ** iteration)) + EPSILON)
        mean = sum(theta) / size
        theta = [value - mean for value in theta]
        if abs(last_loss - loss) <= LOSS_TOLERANCE and max_gradient <= STALL_GRADIENT:
            converged, last_loss = True, loss
            break
        last_loss = loss
    return (dict(zip(candidates, theta)), eta,
            {"converged": converged, "iterations": iteration, "loss": last_loss, "max_gradient": max_gradient})


def strengths(theta: dict[str, float]) -> dict[str, float]:
    scaled = {candidate: math.exp(max(-50.0, min(50.0, value))) for candidate, value in theta.items()}
    total = sum(scaled.values())
    return {candidate: value / total for candidate, value in scaled.items()}


def tiers(strength: dict[str, float], tolerance: float = TIER_TOLERANCE) -> list[list[str]]:
    rows: list[list[str]] = []
    for candidate in sorted(strength, key=lambda name: (-strength[name], name)):
        if rows and abs(strength[rows[-1][0]] - strength[candidate]) <= tolerance:
            rows[-1].append(candidate)
        else:
            rows.append([candidate])
    return rows


def aggregate(groups: list[dict], tolerance: float = TIER_TOLERANCE) -> dict:
    """Strengths, tiers, hard-invalid candidates and per-candidate statistics of deblinded group
    rankings ({"candidates", "tiers", "hard_invalid"}). A hard-invalid flag from any group counts."""
    candidates = sorted({candidate for group in groups for candidate in group["candidates"]})
    observations = pairs(groups)
    theta, eta, diagnostics = fit(candidates, observations)
    strength = strengths(theta)
    statistics = {candidate: {"appearances": 0, "distinct_opponents": 0, "wins": 0, "ties": 0, "losses": 0}
                  for candidate in candidates}
    for group in groups:
        for candidate in group["candidates"]:
            statistics[candidate]["appearances"] += 1
    for candidate in candidates:
        statistics[candidate]["distinct_opponents"] = len({other for group in groups if candidate in group["candidates"]
                                                           for other in group["candidates"] if other != candidate})
    for left, right, outcome in observations:
        if outcome == 2:
            statistics[left]["ties"] += 1
            statistics[right]["ties"] += 1
        else:
            winner, loser = (left, right) if outcome == 0 else (right, left)
            statistics[winner]["wins"] += 1
            statistics[loser]["losses"] += 1
    return {"strengths": strength, "log_strengths": theta, "tie": math.exp(max(-50.0, min(50.0, eta))),
            "fit": diagnostics, "statistics": statistics, "tiers": tiers(strength, tolerance),
            "hard_invalid": sorted({candidate for group in groups for candidate in group["hard_invalid"]})}
