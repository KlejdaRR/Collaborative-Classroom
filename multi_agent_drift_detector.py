# multi_agent_drift_detector.py (FIXED VERSION)
import torch
import numpy as np
from collections import deque
from RuptureDetector import HamiltonianRuptureDetector


class MultiAgentHamiltonianDetector:
    """
    Multi-agent system using Hamiltonian dynamics for robust drift detection.

    Based on the Collaborative Classroom concept where multiple agents
    with different costate dynamics collectively detect concept drift.
    """

    def __init__(self, input_dim=5, num_agents=5, hidden_dim=32):
        self.num_agents = num_agents
        self.input_dim = input_dim

        # Create diverse agents (different decay rates and architectures)
        self.agents = []
        decay_rates = [0.95, 0.98, 0.99, 0.995, 0.999]
        hidden_dims = [hidden_dim, hidden_dim * 2, hidden_dim, hidden_dim * 2, hidden_dim]

        for i in range(min(num_agents, len(decay_rates))):
            agent = HamiltonianRuptureDetector(
                input_dim=input_dim,
                hidden_dim=hidden_dims[i % len(hidden_dims)],
                costate_decay=decay_rates[i % len(decay_rates)]
            )
            self.agents.append(agent)

        # If we need more agents than decay rates, repeat
        while len(self.agents) < num_agents:
            self.agents.append(HamiltonianRuptureDetector(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                costate_decay=0.99
            ))

        # Ensemble voting mechanism
        self.drift_votes = deque(maxlen=100)
        self.agent_weights = [1.0] * num_agents  # Adaptive weights
        self.detection_history = []

        # Meta-learning for weight adaptation
        self.meta_lr = 0.01

    def ensemble_predict(self, x):
        """Ensemble prediction with weighted voting"""
        predictions = []
        for agent in self.agents:
            pred = agent.forward(x)
            pred_val = pred.item() if hasattr(pred, 'item') else pred
            predictions.append(pred_val)

        # Weighted average
        weighted_pred = sum(w * p for w, p in zip(self.agent_weights, predictions))
        weighted_pred /= (sum(self.agent_weights) + 1e-8)

        return weighted_pred

    def update_agent_weights(self, recent_performance):
        """Update agent weights based on recent detection accuracy"""
        total = sum(recent_performance)
        if total > 0:
            self.agent_weights = [p / total for p in recent_performance]

    def detect_drift_ensemble(self, x, y_true, learning_rate=0.005, dt=0.01):
        """Collective drift detection using all agents"""

        # Individual agent updates
        agent_outputs = []
        agent_losses = []
        agent_costates = []

        for agent in self.agents:
            loss, pred = agent.hamiltonian_step(x, y_true, learning_rate, dt)
            pred_val = pred.item() if hasattr(pred, 'item') else pred
            agent_outputs.append(pred_val)
            agent_losses.append(loss)
            agent_costates.append(abs(agent.costate.item()))

        # Detect drift based on costate variance ACROSS agents
        # High variance indicates disagreement -> potential drift
        costate_variance = float(np.var(agent_costates))
        costate_mean = float(np.mean(agent_costates))

        # Ensemble prediction
        ensemble_pred = self.ensemble_predict(x)

        # Drift detection signal
        # Key insight from Hamiltonian theory: Costate divergence indicates
        # that the system is far from optimal trajectory
        drift_signal = costate_variance * costate_mean

        # Classify drift type based on agent consensus
        high_costate_threshold = 0.1
        high_costate_agents = sum(1 for c in agent_costates if c > high_costate_threshold)
        consensus_ratio = high_costate_agents / self.num_agents

        if consensus_ratio > 0.7:
            drift_type = "sudden"  # Most agents agree on drift
        elif consensus_ratio > 0.3:
            drift_type = "gradual"  # Partial agreement
        else:
            drift_type = "none"

        # Adaptive threshold based on recent history
        detection_threshold = 0.05
        if len(self.detection_history) > 50:
            recent_signals = [h['drift_signal'] for h in self.detection_history[-50:]]
            detection_threshold = np.percentile(recent_signals, 80)

        # Record detection
        is_drift = drift_signal > detection_threshold
        self.drift_votes.append(1 if is_drift else 0)
        self.detection_history.append({
            'step': len(self.detection_history),
            'drift_signal': drift_signal,
            'drift_type': drift_type,
            'drift_detected': is_drift,
            'costate_variance': costate_variance,
            'agent_costates': agent_costates,
            'ensemble_pred': ensemble_pred,
            'agent_agreement': consensus_ratio
        })

        return {
            'drift_detected': is_drift,
            'drift_type': drift_type,
            'drift_signal': drift_signal,
            'ensemble_pred': ensemble_pred,
            'agent_agreement': consensus_ratio,
            'agent_costates': agent_costates
        }

    def get_agent_diversity_metrics(self):
        """Measure how diverse the agents' costate dynamics are"""
        if len(self.detection_history) < 10:
            return {'diversity': 0.0, 'stability': 0.0, 'avg_agreement': 0.0}

        recent = self.detection_history[-100:]
        costate_std = [np.std(h['agent_costates']) for h in recent]

        return {
            'diversity': float(np.mean(costate_std)),
            'stability': float(1.0 - np.std(costate_std)),
            'avg_agreement': float(np.mean([h['agent_agreement'] for h in recent]))
        }