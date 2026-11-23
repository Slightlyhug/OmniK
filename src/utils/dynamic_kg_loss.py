import torch
import torch.nn as nn
from typing import Dict, Iterable, Optional, Tuple


class DynamicKGLoss(nn.Module):
    """
    Multi-task loss that combines:
    1. Asymmetric classification loss
    2. Homoscedastic task uncertainty weighting
    3. Knowledge-graph priors with optional dynamic batch co-occurrence updates

    The "dynamic" part is implemented as an exponential moving average over
    observed label co-occurrence statistics, which can be fused with fixed
    biological priors at training time.
    """

    DEFAULT_PRIORS: Tuple[Tuple[str, str, float], ...] = ()

    def __init__(
        self,
        num_tasks: int = 6,
        ptm_types: Optional[Iterable[str]] = None,
        gamma_neg: float = 3.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        kg_weight: float = 0.1,
        dynamic_weight: float = 0.5,
        update_momentum: float = 0.9,
        enable_dynamic_updates: bool = True,
        pos_weight: Optional[torch.Tensor] = None,
        task_weights: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.ptm_types = list(ptm_types or ["ubiquitination", "sumoylation", "succinylation", "methylation", "crotonylation", "acetylation"])
        if len(self.ptm_types) != self.num_tasks:
            raise ValueError(f"ptm_types length {len(self.ptm_types)} does not match num_tasks {self.num_tasks}")
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(max(0.0, clip))
        self.eps = float(max(1e-12, eps))
        self.kg_weight = float(max(0.0, kg_weight))
        self.dynamic_weight = float(max(0.0, min(1.0, dynamic_weight)))
        self.update_momentum = float(max(0.0, min(0.9999, update_momentum)))
        self.enable_dynamic_updates = bool(enable_dynamic_updates)

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.detach().float().view(1, -1))
        else:
            self.pos_weight = None

        if task_weights is None:
            task_weights = [1.0] * self.num_tasks
        weights = torch.as_tensor(list(task_weights), dtype=torch.float32)
        if weights.numel() != self.num_tasks:
            weights = torch.ones(self.num_tasks, dtype=torch.float32)
        self.register_buffer("task_weights", weights.clamp(min=1e-4).view(-1))

        # Dynamic Weights (log_vars) for Homoscedastic Uncertainty.
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

        # Prior graph and its dynamic refinement.
        self.register_buffer("kg_adj", torch.zeros(self.num_tasks, self.num_tasks))
        self.register_buffer("dynamic_adj", torch.zeros(self.num_tasks, self.num_tasks))
        self.register_buffer("running_cooccurrence", torch.zeros(self.num_tasks, self.num_tasks))
        self.register_buffer("running_marginals", torch.zeros(self.num_tasks))
        self.register_buffer("running_support", torch.tensor(0.0))
        self.last_forward_stats: Dict[str, float] = {}
        self._build_kg()

    def _build_kg(self) -> None:
        priors: Dict[Tuple[int, int], float] = {}
        name_to_idx = {str(name): idx for idx, name in enumerate(self.ptm_types)}
        for src_name, dst_name, weight in self.DEFAULT_PRIORS:
            src = name_to_idx.get(src_name)
            dst = name_to_idx.get(dst_name)
            if src is None or dst is None or src == dst:
                continue
            priors[(src, dst)] = float(weight)
            priors[(dst, src)] = float(weight)
        for (src, dst), weight in priors.items():
            self.kg_adj[src, dst] = weight

    def _validate_inputs(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.ndim != 2 or targets.ndim != 2:
            raise ValueError(
                f"logits and targets must be 2D [batch, num_tasks], got logits={tuple(logits.shape)} targets={tuple(targets.shape)}"
            )
        if logits.shape != targets.shape:
            raise ValueError(f"logits shape {tuple(logits.shape)} must match targets shape {tuple(targets.shape)}")
        if logits.shape[1] != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} tasks, got {logits.shape[1]}")
        return logits.float(), targets.float()

    def _update_dynamic_graph(self, targets: torch.Tensor) -> None:
        if not self.enable_dynamic_updates or targets.numel() == 0:
            return
        with torch.no_grad():
            batch_size = max(1, int(targets.shape[0]))
            batch_coocc = torch.matmul(targets.transpose(0, 1), targets) / float(batch_size)
            batch_coocc = batch_coocc.clamp(0.0, 1.0)
            batch_marginals = targets.mean(dim=0).clamp(0.0, 1.0)
            diag = torch.diag(torch.diag(batch_coocc))
            batch_coocc = batch_coocc - diag
            self.running_cooccurrence.mul_(self.update_momentum).add_(batch_coocc * (1.0 - self.update_momentum))
            self.running_marginals.mul_(self.update_momentum).add_(batch_marginals * (1.0 - self.update_momentum))
            self.running_support.mul_(self.update_momentum).add_(1.0 - self.update_momentum)

            expected = torch.outer(self.running_marginals, self.running_marginals)
            dynamic_adj = self.running_cooccurrence - expected
            dynamic_adj = dynamic_adj / 0.25
            dynamic_adj.fill_diagonal_(0.0)
            dynamic_adj = dynamic_adj.clamp(-1.0, 1.0)
            self.dynamic_adj.copy_(dynamic_adj)

    def get_effective_adjacency(self) -> torch.Tensor:
        return self.kg_adj + self.dynamic_adj * self.dynamic_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, targets = self._validate_inputs(logits, targets)
        probs = torch.sigmoid(logits)

        if self.training:
            self._update_dynamic_graph(targets.detach())

        # 1. Asymmetric Loss (Base Loss)
        probs_pos = probs.clamp(min=self.eps, max=1.0 - self.eps)
        probs_neg = (1.0 - probs).clamp(min=self.eps, max=1.0 - self.eps)

        # Hard clipping for easy negatives
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(min=self.eps, max=1.0)

        loss_pos = -torch.pow(1.0 - probs_pos, self.gamma_pos) * torch.log(probs_pos)
        loss_neg = -torch.pow(1.0 - probs_neg, self.gamma_neg) * torch.log(probs_neg.clamp(min=self.eps))

        if self.pos_weight is not None:
            loss_pos = loss_pos * self.pos_weight.to(loss_pos.device)

        base_loss = targets * loss_pos + (1.0 - targets) * loss_neg
        base_loss = base_loss.mean(dim=0)  # [num_tasks]

        weighted_loss = base_loss * self.task_weights.to(base_loss.device)

        # 2. Dynamic Weighting (Uncertainty)
        precision = torch.exp(-self.log_vars)
        dynamic_loss = torch.sum(weighted_loss * precision + 0.5 * self.log_vars)

        # 3. Knowledge Graph Constraints
        effective_adj = self.get_effective_adjacency()
        kg_penalty = logits.new_zeros(())
        for i in range(self.num_tasks):
            for j in range(i + 1, self.num_tasks):
                relation_weight = float(effective_adj[i, j].item())
                if abs(relation_weight) < 1e-8:
                    continue
                co_prob = probs[:, i] * probs[:, j]
                if relation_weight < 0:
                    kg_penalty = kg_penalty + co_prob.mean() * abs(relation_weight)
                else:
                    kg_penalty = kg_penalty + (1.0 - co_prob).mean() * relation_weight

        with torch.no_grad():
            self.last_forward_stats = {
                "dynamic_kg_dynamic_loss": float(dynamic_loss.detach().cpu().item()),
                "dynamic_kg_penalty": float(kg_penalty.detach().cpu().item()),
                "dynamic_kg_log_var_mean": float(self.log_vars.detach().mean().cpu().item()),
                "dynamic_kg_prior_adj_abs_mean": float(self.kg_adj.detach().abs().mean().cpu().item()),
                "dynamic_kg_dynamic_adj_abs_mean": float(self.dynamic_adj.detach().abs().mean().cpu().item()),
                "dynamic_kg_running_support": float(self.running_support.detach().cpu().item()),
                "dynamic_kg_task_weight_mean": float(self.task_weights.detach().mean().cpu().item()),
                "dynamic_kg_pos_weight_mean": float(self.pos_weight.detach().mean().cpu().item()) if self.pos_weight is not None else float("nan"),
            }

        total_loss = dynamic_loss + self.kg_weight * kg_penalty
        return total_loss
