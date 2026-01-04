# documents_scorer_improved.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaV2Model

class LabelAwarePooler(nn.Module):
    """
    对 encoder 的 token 隐藏做 label-aware attention pooling。
    - num_tasks: 3 (coverage, utility, depth)
    - task_hidden: 每个任务内部的表示维度（通常等于 backbone hidden）
    """
    def __init__(self, hidden_size, num_tasks=3, dropout=0.1):
        super().__init__()
        self.num_tasks = num_tasks
        self.hidden_size = hidden_size
        # 每个 task 一个可学习的 query 向量 (1, hidden)
        # 使用多头效果可以扩展成 (num_heads, hidden)
        self.task_queries = nn.Parameter(torch.randn(num_tasks, hidden_size) * 0.02)
        # 一个可共享的投影将 encoder hidden 投影到 attention 空间（可选）
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: (batch, seq_len, hidden)
        attention_mask: (batch, seq_len)
        returns: (batch, num_tasks, hidden)
        """
        keys = self.key_proj(hidden_states)  # (B, L, H)
    
        # 计算每个 task query 对每个 token 的注意力分数
        attn_scores = torch.matmul(keys, self.task_queries.t())  # (B, L, T)
        attn_scores = attn_scores.transpose(1, 2)  # (B, T, L)
    
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)  # (B,1,L)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-1e9"))
    
        attn_probs = torch.softmax(attn_scores, dim=-1)  # (B, T, L)
        pooled = torch.matmul(attn_probs, hidden_states)  # (B, T, H)
        pooled = self.activation(pooled)
        pooled = self.dropout(pooled)
        return pooled  # (B, T, H)



class TaskHead(nn.Module):
    """
    每个任务一个小型分类 head：可选的 bottleneck -> classifier
    输出 logits for binary classification (2 classes)
    """
    def __init__(self, hidden_size, inner_dim=256, dropout=0.1, num_labels=2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, inner_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(inner_dim, num_labels)

    def forward(self, x):
        # x: (batch, hidden)
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        logits = self.classifier(h)
        return logits


class DebertaForMultiHeadClassification(nn.Module):
    """
    Improved model:
    - Deberta backbone
    - LabelAwarePooler producing per-task pooled vectors
    - Per-task small MLP head
    - Per-task class-weighted CrossEntropyLoss if class_weights provided
    """
    def __init__(self, model_name="microsoft/deberta-v3-base", hidden_size_override=None, class_weights=None, dropout=0.1):
        super().__init__()
        self.backbone = DebertaV2Model.from_pretrained(model_name)
        hidden_size = hidden_size_override or self.backbone.config.hidden_size
        self.num_tasks = 3  # coverage, utility, depth
        self.pooler = LabelAwarePooler(hidden_size, num_tasks=self.num_tasks, dropout=dropout)
        # per-task heads
        self.coverage_head = TaskHead(hidden_size, inner_dim=max(128, hidden_size // 4), dropout=dropout, num_labels=2)
        self.utility_head  = TaskHead(hidden_size, inner_dim=max(128, hidden_size // 4), dropout=dropout, num_labels=2)
        self.depth_head    = TaskHead(hidden_size, inner_dim=max(128, hidden_size // 4), dropout=dropout, num_labels=2)

        # class weights: expect dict with "coverage","utility","depth" tensors on device when passed to loss
        self.class_weights = class_weights
        # define losses
        self.loss_fns = {
            "coverage": nn.CrossEntropyLoss(weight=self.class_weights["coverage"]) if self.class_weights and "coverage" in self.class_weights else nn.CrossEntropyLoss(),
            "utility" : nn.CrossEntropyLoss(weight=self.class_weights["utility"]) if self.class_weights and "utility" in self.class_weights else nn.CrossEntropyLoss(),
            "depth"   : nn.CrossEntropyLoss(weight=self.class_weights["depth"]) if self.class_weights and "depth" in self.class_weights else nn.CrossEntropyLoss(),
        }

    def forward(self, input_ids, attention_mask, labels=None):
        """
        labels: (batch,3) long tensor with 0/1 per task, order [coverage, utility, depth]
        returns: dict with logits and per-task losses if labels provided
        """
        b = input_ids.size(0)
        enc = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = enc.last_hidden_state  # (B, L, H)

        # Label-aware pooling -> (B, T, H)
        pooled = self.pooler(hidden, attention_mask=attention_mask)  # (B, 3, H)

        # per-task pooled vectors
        cov_vec = pooled[:, 0, :]
        utl_vec = pooled[:, 1, :]
        dep_vec = pooled[:, 2, :]

        cov_logits = self.coverage_head(cov_vec)  # (B, 2)
        utl_logits = self.utility_head(utl_vec)
        dep_logits = self.depth_head(dep_vec)

        output = {
            "coverage_logits": cov_logits,
            "utility_logits": utl_logits,
            "depth_logits": dep_logits
        }

        loss = None
        if labels is not None:
            cov_labels = labels[:, 0]
            utl_labels = labels[:, 1]
            dep_labels = labels[:, 2]

            cov_loss = self.loss_fns["coverage"](cov_logits, cov_labels)
            utl_loss = self.loss_fns["utility"](utl_logits, utl_labels)
            dep_loss = self.loss_fns["depth"](dep_logits, dep_labels)

            # 可调的 loss 权重（如果需要可以在训练脚本里调整）
            total_loss = (cov_loss + utl_loss + dep_loss) / 3.0

            output["coverage_loss"] = cov_loss
            output["utility_loss"] = utl_loss
            output["depth_loss"] = dep_loss
            output["loss"] = total_loss

        return output
