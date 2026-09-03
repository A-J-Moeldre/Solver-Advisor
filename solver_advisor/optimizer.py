# solver_advisor/optimizer.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math
import random


@dataclass
class ArmStats:
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.total_reward / self.pulls


class BanditOptimizer:
    """
    Simple multi-armed bandit over solver configurations.

    - Each config is an arm.
    - Reward = -runtime or -iterations (lower is better).
    - UCB selection with optional exploration budget.
    """

    def __init__(
        self,
        configs: List[Dict[str, Any]],
        exploration_budget: float = 0.05,
        ucb_alpha: float = 1.0,
    ) -> None:
        if not configs:
            raise ValueError("BanditOptimizer requires at least one config.")

        self.configs: List[Dict[str, Any]] = configs
        self.n_arms: int = len(configs)
        self.stats: List[ArmStats] = [ArmStats() for _ in configs]

        self.total_pulls: int = 0
        self.exploration_budget: float = exploration_budget
        self.ucb_alpha: float = ucb_alpha

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def select_config(self) -> Dict[str, Any]:
        """
        Select a configuration to try next.

        Strategy:
        - If some arms have never been pulled → explore them first.
        - Otherwise, use UCB on mean rewards.
        - Exploration budget: with small probability, pick a random arm.
        """
        self.total_pulls += 1

        # 1. Ensure each arm is tried at least once
        for idx, stats in enumerate(self.stats):
            if stats.pulls == 0:
                return self.configs[idx]

        # 2. Exploration vs exploitation
        if self._should_explore():
            idx = random.randrange(self.n_arms)
            return self.configs[idx]

        # 3. UCB selection
        idx = self._select_ucb_arm()
        return self.configs[idx]

    def update(self, config: Dict[str, Any], performance: Dict[str, Any]) -> None:
        """
        Update bandit state after a solve.

        performance:
            {
                "runtime": float (seconds) OR
                "iterations": int
            }

        Reward is defined as:
            reward = -runtime  (if available)
            reward = -iterations (fallback)
        """
        arm_idx = self._find_arm_index(config)
        if arm_idx is None:
            # Config not in the original pool; ignore for now.
            return

        reward = self._compute_reward(performance)
        stats = self.stats[arm_idx]
        stats.pulls += 1
        stats.total_reward += reward

    # --------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------

    def _compute_reward(self, perf: Dict[str, Any]) -> float:
        if "runtime" in perf:
            return -float(perf["runtime"])
        if "iterations" in perf:
            return -float(perf["iterations"])
        # Unknown performance → neutral reward
        return 0.0

    def _find_arm_index(self, config: Dict[str, Any]) -> Optional[int]:
        # Simple identity-based match; you can later replace with IDs.
        for idx, cfg in enumerate(self.configs):
            if cfg is config:
                return idx
            # or structural equality:
            if cfg == config:
                return idx
        return None

    def _should_explore(self) -> bool:
        """
        Exploration budget as a probability per pull.

        Example:
            exploration_budget = 0.05 → 5% of selections are random.
        """
        if self.exploration_budget <= 0.0:
            return False
        return random.random() < self.exploration_budget

    def _select_ucb_arm(self) -> int:
        """
        Upper Confidence Bound (UCB1-like):

            UCB_i = mean_reward_i + alpha * sqrt(2 * ln(T) / pulls_i)

        where:
            T = total pulls so far
        """
        T = max(self.total_pulls, 1)
        best_idx = 0
        best_ucb = -float("inf")

        for idx, stats in enumerate(self.stats):
            if stats.pulls == 0:
                # Should have been caught earlier, but keep safe.
                return idx

            mean = stats.mean_reward
            bonus = self.ucb_alpha * math.sqrt(2.0 * math.log(T) / stats.pulls)
            ucb = mean + bonus

            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = idx

        return best_idx
