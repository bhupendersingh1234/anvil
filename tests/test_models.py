import json

import pytest

from anvil.models import baselines
from anvil.models.deep_neural_network import DeepNeuralNetwork, DeepNeuralNetworkPricer, DeepNeuralNetworkTrainer
from anvil.models.finetune_openai import make_jsonl
from anvil.models.frontier import messages_for
from anvil.models.neural_network import neural_network_pricer


@pytest.mark.parametrize("name", ["random", "constant", "linear", "nlp-linear", "random-forest", "xgboost"])
def test_baselines_fit_and_predict(name, items) -> None:
    predict = baselines.BASELINES[name](items[:250])
    assert all(float(predict(item)) >= 0 for item in items[250:260])


def test_human_pricer_roundtrip(tmp_path, items) -> None:
    path = tmp_path / "human.csv"
    baselines.write_human_input(items, path, size=3)
    assert baselines.human_pricer(path)(items[1]) == 0


def test_simple_neural_network_trains(items) -> None:
    predict = neural_network_pricer(items, epochs=1, n_features=64)
    assert predict(items[0]) >= 0


def test_dnn_train_save_load_roundtrip(tmp_path, items) -> None:
    dnn = DeepNeuralNetworkPricer(DeepNeuralNetwork(num_layers=3, hidden_size=32))
    mae = DeepNeuralNetworkTrainer(dnn, items[:250], items[250:]).train(epochs=1)
    assert mae >= 0
    path = tmp_path / "dnn.pth"
    dnn.save(path)
    loaded = DeepNeuralNetworkPricer(DeepNeuralNetwork(num_layers=3, hidden_size=32))
    loaded.load(path)
    assert loaded.y_mean == pytest.approx(dnn.y_mean)
    assert loaded(items[0]) == pytest.approx(dnn(items[0]), rel=1e-4)


def test_dnn_default_architecture_matches_published_weights() -> None:
    keys = set(DeepNeuralNetwork(hidden_size=8).state_dict())
    assert {"input_layer.0.weight", "residual_blocks.7.block.4.bias", "output_layer.weight"} <= keys
    assert not any(key.startswith("residual_blocks.8") for key in keys)


def test_openai_finetune_jsonl(items) -> None:
    rows = [json.loads(line) for line in make_jsonl(items[:3]).splitlines()]
    assert rows[0]["messages"][0] == messages_for(items[0].summary)[0]
    assert rows[0]["messages"][1] == {"role": "assistant", "content": f"${items[0].price:.2f}"}
