import pytest

from anvil import cli
from anvil.data import store


def test_parser_accepts_every_command() -> None:
    parser = cli.parser()
    assert parser.parse_args(["evaluate", "xgboost", "--full"]).lite is False
    assert parser.parse_args(["summarize", "submit"]).mode == "submit"
    assert parser.parse_args(["run", "--autonomous"]).autonomous
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "not-a-model"])


@pytest.mark.parametrize("model", ["constant", "linear", "xgboost"])
def test_evaluate_end_to_end_on_local_data(tmp_path, monkeypatch, capsys, items, model) -> None:
    monkeypatch.setattr(store.config, "DATA_DIR", tmp_path)
    store.save_splits("items", True, (items[:250], items[250:270], items[270:]))
    cli.main(["evaluate", model, "--size", "20"])
    assert f"{model}: error=$" in capsys.readouterr().out


def test_train_dnn_command(tmp_path, monkeypatch, items) -> None:
    monkeypatch.setattr(store.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr("anvil.models.deep_neural_network.DeepNeuralNetwork.__init__.__defaults__", (5000, 3, 16, 0.2))
    store.save_splits("items", True, (items[:250], items[250:270], items[270:]))
    weights = tmp_path / "dnn.pth"
    cli.main(["train-dnn", "--epochs", "1", "--weights", str(weights)])
    cli.main(["evaluate", "dnn", "--size", "10", "--weights", str(weights)])
    assert weights.exists() and weights.with_suffix(".json").exists()


def test_curate_uses_cache_and_caps_per_category(tmp_path, monkeypatch, capsys, items) -> None:
    monkeypatch.setattr(store.config, "DATA_DIR", tmp_path)
    loaded = []
    monkeypatch.setattr(
        "anvil.data.loaders.load_category",
        lambda category, workers: (
            loaded.append(category)
            or [item.model_copy(update={"category": category, "title": f"{category} {item.title}", "full": f"{category} {item.full}"}) for item in items]
        ),
    )
    cli.main(["curate", "--categories", "Appliances", "Toys_and_Games", "--per-category", "100", "--sample-size", "150"])
    cli.main(["curate", "--categories", "Appliances", "Toys_and_Games", "--per-category", "100", "--sample-size", "150"])
    assert loaded == ["Appliances", "Toys_and_Games"]
    train, val, test = store.load_splits("items_raw", False)
    assert len(train) + len(val) + len(test) == 150
    assert "Kept a random 100 Appliances items" in capsys.readouterr().out 