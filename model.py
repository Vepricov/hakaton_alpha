"""
FT-Transformer (Feature Tokenizer Transformer) для табличных данных
Адаптировано для задачи прогнозирования доходов
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional


class FeatureTokenizer(nn.Module):
    """
    Преобразует числовые и категориальные признаки в токены для трансформера
    """
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int = 192,
        embedding_dropout: float = 0.1
    ):
        super().__init__()
        self.n_num_features = n_num_features
        self.n_cat_features = len(cat_cardinalities)
        self.d_token = d_token

        # Для числовых признаков - линейное преобразование
        if n_num_features > 0:
            self.num_tokenizer = nn.Linear(1, d_token)
            # Bias для каждого числового признака
            self.num_bias = nn.Parameter(torch.zeros(n_num_features, d_token))

        # Для категориальных признаков - embedding таблицы
        if self.n_cat_features > 0:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, d_token)
                for cardinality in cat_cardinalities
            ])

        self.dropout = nn.Dropout(embedding_dropout)

    def forward(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]):
        """
        Args:
            x_num: (batch_size, n_num_features) - числовые признаки
            x_cat: (batch_size, n_cat_features) - категориальные признаки (индексы)
        Returns:
            tokens: (batch_size, n_features, d_token)
        """
        tokens = []

        # Обработка числовых признаков
        if x_num is not None and self.n_num_features > 0:
            # (batch, n_num) -> (batch, n_num, 1) -> (batch, n_num, d_token)
            num_tokens = self.num_tokenizer(x_num.unsqueeze(-1))
            num_tokens = num_tokens + self.num_bias.unsqueeze(0)
            tokens.append(num_tokens)

        # Обработка категориальных признаков
        if x_cat is not None and self.n_cat_features > 0:
            cat_tokens = []
            for i, embedding in enumerate(self.cat_embeddings):
                cat_tokens.append(embedding(x_cat[:, i]))
            cat_tokens = torch.stack(cat_tokens, dim=1)  # (batch, n_cat, d_token)
            tokens.append(cat_tokens)

        # Объединяем все токены
        tokens = torch.cat(tokens, dim=1)  # (batch, n_features, d_token)
        tokens = self.dropout(tokens)

        return tokens


class TransformerBlock(nn.Module):
    """
    Стандартный блок трансформера с self-attention и FFN
    """
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        d_ffn: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, n_tokens, d_token)
        Returns:
            out: (batch_size, n_tokens, d_token)
        """
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class FTTransformer(nn.Module):
    """
    FT-Transformer для регрессии доходов
    """
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        d_token: int = 192,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ffn: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.2
    ):
        super().__init__()

        self.tokenizer = FeatureTokenizer(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_token=d_token,
            embedding_dropout=dropout
        )

        # Специальный [CLS] токен для агрегации
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        # Трансформер блоки
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_token=d_token,
                n_heads=n_heads,
                d_ffn=d_ffn,
                dropout=dropout,
                attention_dropout=attention_dropout
            )
            for _ in range(n_layers)
        ])

        # Финальная голова для регрессии
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Инициализация весов"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        x_num: Optional[torch.Tensor] = None,
        x_cat: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ):
        """
        Args:
            x_num: (batch_size, n_num_features)
            x_cat: (batch_size, n_cat_features)
            return_attention: если True, возвращает attention weights для объяснимости
        Returns:
            predictions: (batch_size, 1) - предсказания дохода
            attention_weights: optional, для SHAP
        """
        # Токенизация признаков
        tokens = self.tokenizer(x_num, x_cat)  # (batch, n_features, d_token)

        # Добавляем CLS токен
        batch_size = tokens.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (batch, 1+n_features, d_token)

        # Пропускаем через трансформер
        attention_weights = []
        for block in self.transformer_blocks:
            tokens = block(tokens)
            if return_attention:
                # Для упрощения берем последний слой attention
                pass

        # Используем CLS токен для предсказания
        cls_output = tokens[:, 0, :]  # (batch, d_token)

        # Регрессионная голова
        predictions = self.head(cls_output)  # (batch, 1)

        if return_attention:
            return predictions, attention_weights

        return predictions


class WeightedMAELoss(nn.Module):
    """
    Weighted Mean Absolute Error Loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            predictions: (batch_size, 1)
            targets: (batch_size, 1)
            weights: (batch_size, 1)
        """
        mae = torch.abs(predictions - targets)
        wmae = (weights * mae).mean()
        return wmae


class TabularDataset(torch.utils.data.Dataset):
    """
    Dataset для табличных данных
    """
    def __init__(
        self,
        x_num: Optional[np.ndarray],
        x_cat: Optional[np.ndarray],
        y: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None
    ):
        self.x_num = torch.FloatTensor(x_num) if x_num is not None else None
        self.x_cat = torch.LongTensor(x_cat) if x_cat is not None else None
        self.y = torch.FloatTensor(y).unsqueeze(1) if y is not None else None
        self.weights = torch.FloatTensor(weights).unsqueeze(1) if weights is not None else None

        # Определяем длину
        if self.x_num is not None:
            self.length = len(self.x_num)
        elif self.x_cat is not None:
            self.length = len(self.x_cat)
        else:
            raise ValueError("Either x_num or x_cat must be provided")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {}

        if self.x_num is not None:
            item['x_num'] = self.x_num[idx]

        if self.x_cat is not None:
            item['x_cat'] = self.x_cat[idx]

        if self.y is not None:
            item['y'] = self.y[idx]

        if self.weights is not None:
            item['weights'] = self.weights[idx]

        return item
