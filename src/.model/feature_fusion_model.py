import torch
import torch.nn as nn
from transformers import RobertaModel

class RobertaWithInternalFeatures(nn.Module):
    def __init__(self, roberta_name="roberta-base", feature_dim=6):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained(roberta_name)
        self.feature_extractor = FeatureExtractor()


        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )


        self.classifier = nn.Sequential(
            nn.Linear(768 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask, queries, labels=None):

        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (B, 768)


        features = self.feature_extractor.extract(queries)
        features = features.to(cls_emb.device)
        feat_emb = self.feature_encoder(features)     # (B, 32)


        fused = torch.cat([cls_emb, feat_emb], dim=-1)
        logits = self.classifier(fused)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits

        return logits
