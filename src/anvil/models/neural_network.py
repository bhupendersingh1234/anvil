import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from anvil.data.items import Item
from anvil.models.baselines import Predictor


class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        sizes = [input_size, 128, 64, 64, 64, 64, 64, 64]
        layers = [module for a, b in zip(sizes, sizes[1:]) for module in (nn.Linear(a, b), nn.ReLU())]
        self.layers = nn.Sequential(*layers, nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def neural_network_pricer(train: list[Item], epochs: int = 2, n_features: int = 5000) -> Predictor:
    np.random.seed(42)
    torch.manual_seed(42)
    vectorizer = HashingVectorizer(n_features=n_features, stop_words="english", binary=True)
    X = torch.FloatTensor(vectorizer.fit_transform([item.summary for item in train]).toarray())
    y = torch.FloatTensor([item.price for item in train]).unsqueeze(1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=42)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    model = NeuralNetwork(n_features)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_X, batch_y in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            optimizer.zero_grad()
            loss = loss_function(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            print(f"Epoch {epoch}/{epochs} train loss {loss.item():.3f} val loss {loss_function(model(X_val), y_val).item():.3f}")

    def predict(item: Item) -> float:
        with torch.no_grad():
            return max(0.0, model(torch.FloatTensor(vectorizer.transform([item.summary]).toarray()))[0].item())

    return predict
