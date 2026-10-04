import pytest

from anvil.evaluation import Tester, color_for, evaluate, make_title, running_error, truth_and_title


def test_evaluate_perfect_predictor(items) -> None:
    metrics = evaluate(lambda item: item.price, items, size=50, show=False)
    assert metrics.error == 0 and metrics.r2 == 100


def test_evaluate_parses_string_guesses_and_clips_size(items) -> None:
    tester = Tester(lambda item: f"${item.price + 10:,.2f}", items[:20], title="Plus Ten", size=500)
    metrics = tester.run(verbose=False)
    assert tester.size == 20 and metrics.error == pytest.approx(10)
    assert tester.scatter_chart(metrics) and tester.error_trend_chart(metrics)


def test_dict_datapoints() -> None:
    datapoint = {"prompt": "Question\n\nTitle: A very long product title that goes on and on\nCategory: X\n\nPrice is $", "completion": "12.00"}
    truth, title = truth_and_title(datapoint)
    assert truth == 12 and title.endswith("...") and len(title) == 43


def test_helpers() -> None:
    def gpt_4__1_nano() -> None:
        pass

    assert make_title(gpt_4__1_nano) == "GPT 4.1 Nano"
    assert color_for(10, 100) == "green" and color_for(60, 100) == "orange" and color_for(200, 100) == "red"
    means, ci = running_error([10, 20, 30])
    assert means == [10, 15, 20] and ci[0] == 0 and ci[-1] > 0
