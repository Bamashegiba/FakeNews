import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        hidden_dim=512,
        num_classes=2,
        max_length=256,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        # Поддержка прямого прохода через эмбеддинги
        if inputs_embeds is not None:
            x = inputs_embeds
            batch_size, seq_len, _ = x.size()
        else:
            batch_size, seq_len = input_ids.size()
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
            x = self.embedding(input_ids) + self.position_embedding(positions)

        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        cls_representation = x[:, 0, :]
        cls_representation = self.dropout(cls_representation)
        logits = self.classifier(cls_representation)
        return logits
