import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import HashingVectorizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from anvil.data.items import Item

N_FEATURES = 5000
Y_MEAN = 4.434937953948975
Y_STD = 1.0328539609909058


def vectorizer() -> HashingVectorizer:
    return HashingVectorizer(n_features=N_FEATURES, stop_words="english", binary=True)


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout_prob: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size: int = N_FEATURES, num_layers: int = 10, hidden_size: int = 4096, dropout_prob: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(), nn.Dropout(dropout_prob))
        self.residual_blocks = nn.ModuleList(ResidualBlock(hidden_size, dropout_prob) for _ in range(num_layers - 2))
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x)


class DeepNeuralNetworkPricer:
    def __init__(self, model: DeepNeuralNetwork | None = None, y_mean: float = Y_MEAN, y_std: float = Y_STD):
        seed_everything()
        self.device = best_device()
        self.vectorizer = vectorizer()
        self.model = (model or DeepNeuralNetwork()).to(self.device)
        self.y_mean = y_mean
        self.y_std = y_std
        logging.info(f"Neural Network is using {self.device}")

    def load(self, path: Path) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        stats = Path(path).with_suffix(".json")
        if stats.exists():
            data = json.loads(stats.read_text())
            self.y_mean, self.y_std = data["y_mean"], data["y_std"]

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)
        Path(path).with_suffix(".json").write_text(json.dumps({"y_mean": self.y_mean, "y_std": self.y_std}))

    def encode(self, texts: list[str]) -> torch.Tensor:
        return torch.FloatTensor(self.vectorizer.transform(texts).toarray())

    def price(self, text: str) -> float:
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.encode([text]).to(self.device))[0]
            return max(0.0, (torch.exp(pred * self.y_std + self.y_mean) - 1).item())

    def __call__(self, item: Item) -> float:
        return self.price(item.summary)


class DeepNeuralNetworkTrainer:
    def __init__(self, dnn: DeepNeuralNetworkPricer, train: list[Item], val: list[Item], batch_size: int = 64):
        self.dnn = dnn
        self.X_val = dnn.encode([item.summary for item in val])
        self.y_val = torch.FloatTensor([item.price for item in val]).unsqueeze(1)
        y_train_log = torch.log(torch.FloatTensor([item.price for item in train]).unsqueeze(1) + 1)
        dnn.y_mean, dnn.y_std = y_train_log.mean().item(), y_train_log.std().item()
        self.y_val_norm = (torch.log(self.y_val + 1) - dnn.y_mean) / dnn.y_std
        y_train_norm = (y_train_log - dnn.y_mean) / dnn.y_std
        X_train = dnn.encode([item.summary for item in train])
        self.loader = DataLoader(TensorDataset(X_train, y_train_norm), batch_size=batch_size, shuffle=True)
        self.loss_function = nn.L1Loss()
        self.optimizer = torch.optim.AdamW(dnn.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
        params = sum(p.numel() for p in dnn.model.parameters() if p.requires_grad)
        print(f"Deep Neural Network created with {params:,} parameters on {dnn.device}")

    def train(self, epochs: int = 5) -> float:
        model, device = self.dnn.model, self.dnn.device
        mae = float("nan")
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for batch_X, batch_y in tqdm(self.loader, desc=f"Epoch {epoch}/{epochs}"):
                self.optimizer.zero_grad()
                loss = self.loss_function(model(batch_X.to(device)), batch_y.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()
                losses.append(loss.item())
            model.eval()
            with torch.no_grad():
                outputs = model(self.X_val.to(device))
                val_loss = self.loss_function(outputs, self.y_val_norm.to(device)).item()
                prices = torch.exp(outputs * self.dnn.y_std + self.dnn.y_mean) - 1
                mae = torch.abs(prices - self.y_val.to(device)).mean().item()
            print(
                f"Epoch {epoch}/{epochs} train loss {np.mean(losses):.4f} val loss {val_loss:.4f} val MAE ${mae:.2f} lr {self.scheduler.get_last_lr()[0]:.6f}"
            )
            self.scheduler.step()
        return mae
