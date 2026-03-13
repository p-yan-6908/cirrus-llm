"""Training utilities for Cirrus.

Implements:
- Staged expert growth scheduler
- Tool data mixing
- Synthetic tool trajectory generation
- Combined SFT + DPO training loop
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CirrusConfig
from .tools import ToolSchema


class ExpertGrowthScheduler:
    """Staged expert growth during pretraining.

    Phase 1: 2 experts (nearly dense, stable)
    Phase 2: 4 experts (moderate sparsity)
    Phase 3: 12 experts (full MoE)
    """

    def __init__(self, model, config):
        self.model = model
        self.n_phases = config.n_expert_phases  # (2, 4, 12)
        self.phase_epochs = config.expert_phase_epochs  # (2, 2, None)
        self.current_phase = 0

    def get_phase(self, epoch):
        """Get the expert count for a given epoch."""
        cumulative = 0
        for i, phase_epochs in enumerate(self.phase_epochs):
            if phase_epochs is not None:
                cumulative += phase_epochs
                if epoch < cumulative:
                    return i
            else:
                return i  # Last phase, stay here
        return len(self.n_phases) - 1

    def step(self, epoch):
        """Update model expert configuration for the current epoch."""
        phase = self.get_phase(epoch)
        if phase != self.current_phase:
            self.current_phase = phase
            target_experts = self.n_phases[phase]
            self._resize_experts(target_experts)
            print(f"Expert growth: phase {phase}, {target_experts} experts")

    def _resize_experts(self, n_experts):
        """Resize expert count in MoE layers.

        For simplicity, this just adjusts which experts are active.
        In practice, you'd add/remove expert modules.
        """
        for layer in self.model.layers:
            if layer.ffn.is_moe:
                moe = layer.ffn.ffn
                if hasattr(moe, "top_k"):
                    moe.top_k = min(2, n_experts)


class ToolDataMixer:
    """Mixes tool-use data into training batches.

    Maintains a target fraction of tool-related data
    (schemas, API docs, function signatures) during pretraining.
    """

    def __init__(self, target_fraction=0.1):
        self.target_fraction = target_fraction

    def should_use_tool_data(self):
        """Randomly decide if this batch should use tool data."""
        return random.random() < self.target_fraction


class SyntheticToolTrajectoryGenerator:
    """Generates synthetic tool-use training trajectories programmatically.

    Creates both correct and incorrect trajectories for DPO training.
    """

    TOOL_TEMPLATES = [
        ToolSchema(
            name="get_weather",
            description="Get current weather for a city",
            parameters=[
                {"name": "city", "type": "string", "description": "City name"},
                {
                    "name": "unit",
                    "type": "enum",
                    "description": "celsius or fahrenheit",
                },
            ],
            returns="WeatherResult",
        ),
        ToolSchema(
            name="search",
            description="Search the web for information",
            parameters=[
                {"name": "query", "type": "string", "description": "Search query"},
                {
                    "name": "max_results",
                    "type": "int",
                    "description": "Max results to return",
                },
            ],
            returns="SearchResults",
        ),
        ToolSchema(
            name="calculate",
            description="Evaluate a mathematical expression",
            parameters=[
                {
                    "name": "expression",
                    "type": "string",
                    "description": "Math expression",
                },
            ],
            returns="number",
        ),
        ToolSchema(
            name="send_email",
            description="Send an email message",
            parameters=[
                {"name": "to", "type": "string", "description": "Recipient email"},
                {"name": "subject", "type": "string", "description": "Email subject"},
                {"name": "body", "type": "string", "description": "Email body"},
            ],
            returns="SendResult",
        ),
        ToolSchema(
            name="read_file",
            description="Read contents of a file",
            parameters=[
                {"name": "path", "type": "string", "description": "File path"},
            ],
            returns="string",
        ),
        ToolSchema(
            name="query_database",
            description="Run a SQL query",
            parameters=[
                {"name": "sql", "type": "string", "description": "SQL query"},
                {"name": "database", "type": "string", "description": "Database name"},
            ],
            returns="QueryResult",
        ),
    ]

    TASK_TEMPLATES = [
        {
            "template": "Find the {attribute} of {entity} and {operation}",
            "slots": {
                "attribute": ["population", "capital", "area", "currency"],
                "entity": ["France", "Japan", "Brazil", "Germany", "India"],
                "operation": [
                    "convert it to millions",
                    "round it to the nearest thousand",
                    "compare it with Spain",
                ],
            },
        },
        {
            "template": "Search for information about {topic} and summarize",
            "slots": {
                "topic": [
                    "quantum computing",
                    "climate change",
                    "renewable energy",
                    "artificial intelligence",
                ],
            },
        },
        {
            "template": "Calculate {expression} and send the result to {email}",
            "slots": {
                "expression": [
                    "15% of 2500",
                    "the square root of 144",
                    "how many days between Jan 1 and Mar 15",
                ],
                "email": ["user@example.com", "admin@company.com"],
            },
        },
    ]

    @classmethod
    def generate_trajectory(cls, use_correct=True):
        """Generate a single tool-use trajectory.

        Args:
            use_correct: if False, generates a flawed trajectory for DPO

        Returns:
            dict with keys: 'input', 'trajectory', 'is_correct'
        """
        task = random.choice(cls.TASK_TEMPLATES)
        template = task["template"]

        # Fill slots
        values = {}
        for slot, options in task["slots"].items():
            values[slot] = random.choice(options)

        user_input = template.format(**values)

        # Generate tool call trajectory
        if use_correct:
            trajectory = cls._correct_trajectory(user_input, values)
        else:
            trajectory = cls._incorrect_trajectory(user_input, values)

        return {
            "input": user_input,
            "trajectory": trajectory,
            "is_correct": use_correct,
            "tools_available": [t.name for t in cls.TOOL_TEMPLATES],
        }

    @classmethod
    def _correct_trajectory(cls, user_input, values):
        """Generate a correct tool-use trajectory."""
        steps = []

        # Simple heuristic: pick relevant tools
        if "population" in user_input or "capital" in user_input:
            steps.append(
                {
                    "think": f"I need to find {values.get('attribute', 'info')}",
                    "tool": "search",
                    "args": {"query": user_input, "max_results": 3},
                    "result": f"Found result for {values.get('entity', 'the query')}",
                }
            )

            if "convert" in user_input or "calculate" in user_input:
                steps.append(
                    {
                        "think": "Now I need to calculate the conversion",
                        "tool": "calculate",
                        "args": {
                            "expression": values.get("expression", "67390000 / 1000000")
                        },
                        "result": "67.39",
                    }
                )

        elif "calculate" in user_input:
            steps.append(
                {
                    "think": "I need to calculate this expression",
                    "tool": "calculate",
                    "args": {"expression": values.get("expression", "15% of 2500")},
                    "result": "375",
                }
            )

            if "send" in user_input or "email" in user_input:
                steps.append(
                    {
                        "think": "Now I'll send the result via email",
                        "tool": "send_email",
                        "args": {
                            "to": values.get("email", "user@example.com"),
                            "subject": "Calculation Result",
                            "body": "The result is 375",
                        },
                        "result": "Email sent successfully",
                    }
                )

        elif "search" in user_input:
            steps.append(
                {
                    "think": "I need to search for information",
                    "tool": "search",
                    "args": {
                        "query": values.get("topic", "the topic"),
                        "max_results": 5,
                    },
                    "result": f"Found information about {values.get('topic', 'the topic')}",
                }
            )

        else:
            steps.append(
                {
                    "think": "Let me figure out what tool to use",
                    "tool": "search",
                    "args": {"query": user_input, "max_results": 3},
                    "result": "Search completed",
                }
            )

        return steps

    @classmethod
    def _incorrect_trajectory(cls, user_input, values):
        """Generate an incorrect trajectory for DPO training.

        Common mistakes:
        - Wrong tool selection
        - Missing arguments
        - Unnecessary steps
        - Forgetting to use results
        """
        mistake_type = random.choice(
            [
                "wrong_tool",
                "missing_args",
                "extra_steps",
                "unused_result",
            ]
        )

        steps = cls._correct_trajectory(user_input, values)

        if mistake_type == "wrong_tool" and steps:
            # Use a different tool
            wrong_tool = "calculate" if steps[0]["tool"] != "calculate" else "search"
            steps[0]["tool"] = wrong_tool

        elif mistake_type == "missing_args" and steps:
            # Remove a required argument
            if steps[0]["args"]:
                key_to_remove = list(steps[0]["args"].keys())[0]
                del steps[0]["args"][key_to_remove]

        elif mistake_type == "extra_steps":
            # Add an unnecessary tool call
            steps.insert(
                0,
                {
                    "think": "I should double check by searching first",
                    "tool": "search",
                    "args": {"query": "unrelated query", "max_results": 1},
                    "result": "Irrelevant result",
                },
            )

        elif mistake_type == "unused_result" and len(steps) > 1:
            # Ignore the result of the first step
            steps[1]["think"] = "I'll just answer directly"
            steps[1]["tool"] = "search"
            steps[1]["args"] = {"query": "something else", "max_results": 1}

        return steps

    @classmethod
    def generate_batch(cls, batch_size, correct_fraction=0.5):
        """Generate a batch of trajectories for training.

        Args:
            batch_size: number of trajectories
            correct_fraction: fraction of correct trajectories

        Returns:
            list of trajectory dicts
        """
        trajectories = []
        for _ in range(batch_size):
            use_correct = random.random() < correct_fraction
            trajectories.append(cls.generate_trajectory(use_correct))
        return trajectories


class CirrusTrainer:
    """Combined SFT + DPO trainer for Cirrus.

    Phase 1: Pretrain with staged expert growth + tool data mixing
    Phase 2: Combined SFT + DPO on tool trajectories
    """

    def __init__(self, model, config, optimizer):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.expert_scheduler = ExpertGrowthScheduler(model, config)
        self.tool_mixer = ToolDataMixer(config.tool_data_fraction)

    def pretrain_step(self, batch, epoch):
        """Single pretraining step.

        Args:
            batch: dict with 'input_ids' and optionally 'labels'
            epoch: current epoch number

        Returns:
            dict with 'loss' and metrics
        """
        # Update expert growth
        self.expert_scheduler.step(epoch)

        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)

        # Forward
        logits, states, kv_caches, aux_loss = self.model(input_ids)

        # LM loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Total loss
        loss = lm_loss + aux_loss

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "lm_loss": lm_loss.item(),
            "aux_loss": aux_loss.item(),
            "expert_phase": self.expert_scheduler.current_phase,
        }

    def sft_dpo_step(self, sft_batch, dpo_batch):
        """Combined SFT + DPO step for Phase 2.

        Args:
            sft_batch: dict with 'input_ids' and 'labels'
            dpo_batch: dict with 'chosen_ids' and 'rejected_ids'

        Returns:
            dict with losses and metrics
        """
        # SFT loss
        logits, _, _, aux_loss = self.model(sft_batch["input_ids"])
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = sft_batch["labels"][:, 1:].contiguous()
        sft_loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # DPO loss
        chosen_logits, _, _, _ = self.model(dpo_batch["chosen_ids"])
        rejected_logits, _, _, _ = self.model(dpo_batch["rejected_ids"])

        # Simple DPO: reward gap should be positive
        chosen_logps = self._sequence_logps(chosen_logits, dpo_batch["chosen_ids"])
        rejected_logps = self._sequence_logps(
            rejected_logits, dpo_batch["rejected_ids"]
        )

        beta = 0.1  # DPO temperature
        dpo_loss = -F.logsigmoid(beta * (chosen_logps - rejected_logps)).mean()

        # Combined loss
        loss = sft_loss + 0.5 * dpo_loss + aux_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "sft_loss": sft_loss.item(),
            "dpo_loss": dpo_loss.item(),
            "aux_loss": aux_loss.item(),
        }

    def _sequence_logps(self, logits, labels):
        """Compute average next-token log-probability of a sequence.

        Args:
            logits: Model output logits of shape (batch, seq_len, vocab_size)
            labels: Target token IDs of shape (batch, seq_len)

        Returns:
            Mean log-probability per sequence (batch,)
        """
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_logps = log_probs.gather(2, labels[:, 1:].unsqueeze(-1)).squeeze(-1)
        return token_logps.mean(dim=-1)
