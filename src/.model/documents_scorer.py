import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Model


class LabelAwarePooler(nn.Module):
    def __init__(self, hidden_size, num_tasks=3, dropout=0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_size = hidden_size

        self.task_queries = nn.Parameter(torch.randn(num_tasks, hidden_size) * 0.02)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask=None):
        keys = self.key_proj(hidden_states)  # (B, L, H)
        attn_scores = torch.matmul(keys, self.task_queries.t())  # (B, L, T)
        attn_scores = attn_scores.transpose(1, 2)  # (B, T, L)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        pooled = torch.matmul(attn_probs, hidden_states)  # (B, T, H)
        pooled = self.activation(pooled)
        pooled = self.dropout(pooled)
        return pooled  # (B, T, H)


class TaskHead(nn.Module):

    def __init__(self, hidden_size, inner_dim=256, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, inner_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(inner_dim, 1)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        logit = self.classifier(h).squeeze(-1)  # (B,)
        return logit


class DebertaForMultiHeadClassification(nn.Module):

    def __init__(
        self,
        model_name="microsoft/deberta-v3-base",
        hidden_size_override=None,
        pos_weights=None,  # dict: {"coverage": w, "utility": w, "depth": w}
        dropout=0.1
    ):
        super().__init__()
        self.backbone = DebertaV2Model.from_pretrained(model_name)
        hidden_size = hidden_size_override or self.backbone.config.hidden_size

        self.num_tasks = 3  # coverage, utility, depth
        self.pooler = LabelAwarePooler(hidden_size, num_tasks=self.num_tasks, dropout=dropout)

        # task heads
        self.coverage_head = TaskHead(hidden_size, inner_dim=max(128, hidden_size // 4), dropout=dropout)
        self.utility_head  = TaskHead(hidden_size, inner_dim=max(128, hidden_size // 4), dropout=dropout)
        self.depth_head    = TaskHead(hidden_size, inner_dim=max(128, hidden_size // 4), dropout=dropout)

        # BCE losses with positive class weights
        self.loss_fns = {
            "coverage": nn.BCEWithLogitsLoss(pos_weight=pos_weights["coverage"]) if pos_weights else nn.BCEWithLogitsLoss(),
            "utility" : nn.BCEWithLogitsLoss(pos_weight=pos_weights["utility"])  if pos_weights else nn.BCEWithLogitsLoss(),
            "depth"   : nn.BCEWithLogitsLoss(pos_weight=pos_weights["depth"])    if pos_weights else nn.BCEWithLogitsLoss(),
        }

        # log(s_T)
        self.log_s = nn.Parameter(torch.zeros(self.num_tasks))

    def forward(self, input_ids, attention_mask, labels=None):
        enc = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden = enc.last_hidden_state  # (B, L, H)

        pooled = self.pooler(hidden, attention_mask)  # (B, 3, H)

        cov_logit = self.coverage_head(pooled[:, 0, :])  # (B,)
        utl_logit = self.utility_head(pooled[:, 1, :])
        dep_logit = self.depth_head(pooled[:, 2, :])

        output = {
            "coverage_logit": cov_logit,
            "utility_logit": utl_logit,
            "depth_logit": dep_logit,
        }

        if labels is not None:
            labels = labels.float()  # BCE requires float

            cov_loss = self.loss_fns["coverage"](cov_logit, labels[:, 0])
            utl_loss = self.loss_fns["utility"](utl_logit, labels[:, 1])
            dep_loss = self.loss_fns["depth"](dep_logit, labels[:, 2])

            task_losses = torch.stack([cov_loss, utl_loss, dep_loss])

            weighted_loss = torch.sum(
                0.5 * torch.exp(-2 * self.log_s) * task_losses + self.log_s
            )

            output.update({
                "coverage_loss": cov_loss,
                "utility_loss": utl_loss,
                "depth_loss": dep_loss,
                "loss": weighted_loss
            })

        return output
