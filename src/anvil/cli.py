import argparse
from pathlib import Path

from anvil import config

MODELS = [
    "random",
    "constant",
    "linear",
    "nlp-linear",
    "random-forest",
    "xgboost",
    "human",
    "nn",
    "dnn",
    "frontier",
    "rag",
    "openai-ft",
    "qlora",
    "specialist",
    "ensemble",
]


def cmd_curate(args) -> None:
    import random

    from anvil.data import curate, store
    from anvil.data.loaders import load_cached_category

    items = []
    for category in args.categories:
        rows = load_cached_category(category, args.workers, args.force)
        if args.per_category and len(rows) > args.per_category:
            rows = random.Random(42).sample(rows, args.per_category)
            print(f"Kept a random {len(rows):,} {category} items (--per-category)")
        items.extend(rows)
    items = curate.deduplicate(items)
    print(f"{len(items):,} items after deduplication")
    splits = curate.split(curate.weighted_sample(items, args.sample_size), config.SPLITS[False])
    print("Saved full splits to", store.save_splits("items_raw", False, splits))
    print("Saved lite splits to", store.save_splits("items_raw", True, store.lite_subset(splits)))
    if args.push:
        print("Pushed", store.push_splits("items_raw", False, splits), store.push_splits("items_raw", True, store.lite_subset(splits)))


def cmd_summarize(args) -> None:
    from anvil.data import store
    from anvil.data.summarize import BatchRunner, Preprocessor, summarize_items

    splits = store.load_splits("items_raw", args.lite)
    items = [item for rows in splits for item in rows]
    if args.mode == "local":
        summarize_items(items, Preprocessor(args.model or config.PREPROCESSOR_MODEL), args.workers)
    else:
        runner = BatchRunner(items, config.BATCH_DIR / ("lite" if args.lite else "full"))
        if args.mode == "submit":
            print(f"Submitted {runner.submit()} batches")
            return
        runner.apply_all()
        print(f"{runner.fetch()} of {len(runner.batches)} batches complete")
    missing = sum(not item.summary for item in items)
    if missing:
        print(f"{missing:,} items still have no summary - run again later")
        return
    for item in items:
        item.full, item.id = None, None
    sizes = [len(rows) for rows in splits]
    summarized = (items[: sizes[0]], items[sizes[0] : sizes[0] + sizes[1]], items[sizes[0] + sizes[1] :])
    print("Saved to", store.save_splits("items", args.lite, summarized))
    if not args.lite:
        store.save_splits("items", True, store.lite_subset(summarized))
    if args.push:
        print("Pushed", store.push_splits("items", args.lite, summarized))


def cmd_prompts(args) -> None:
    from transformers import AutoTokenizer

    from anvil.data import store
    from anvil.data.prompts import build_prompts

    splits = store.load_splits("items", args.lite)
    truncated = build_prompts(*splits, AutoTokenizer.from_pretrained(config.BASE_MODEL), args.cutoff)
    print(f"Truncated {truncated:.1%} of summaries at {args.cutoff} tokens")
    print("Saved to", store.save_splits("items_prompts", args.lite, splits))
    if args.push:
        print("Pushed", store.push_splits("items_prompts", args.lite, splits))


def cmd_index(args) -> None:
    from anvil import rag
    from anvil.data import store

    train = store.load_splits("items", args.lite)[0]
    count = rag.build_index(rag.get_collection(), rag.load_encoder(), train[: args.limit] if args.limit else train)
    print(f"Vector store at {config.VECTORSTORE_DIR} holds {count:,} products")


def prompts_test(lite: bool) -> list[dict]:
    from anvil.data import store

    path = store.local_dir("items_prompts", lite) / "test.jsonl"
    if path.exists():
        return store.read_jsonl(path)
    from datasets import load_dataset

    return list(load_dataset(config.dataset_name("items_prompts", lite), split="test"))


def build_predictor(args, train):
    name = args.model
    if name in ("random", "constant", "linear", "nlp-linear", "random-forest", "xgboost"):
        from anvil.models.baselines import BASELINES

        return BASELINES[name](train)
    if name == "human":
        from anvil.models.baselines import human_pricer

        return human_pricer(config.HUMAN_PREDICTIONS)
    if name == "nn":
        from anvil.models.neural_network import neural_network_pricer

        return neural_network_pricer(train, args.epochs)
    if name == "dnn":
        from anvil.models.deep_neural_network import DeepNeuralNetworkPricer

        dnn = DeepNeuralNetworkPricer()
        dnn.load(args.weights)
        return dnn
    kwargs = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
    if name == "frontier":
        from anvil.models.frontier import frontier_pricer

        return frontier_pricer(args.llm or "openai/gpt-4.1-nano", **kwargs)
    if name == "openai-ft":
        from openai import OpenAI

        from anvil.models.finetune_openai import fine_tuned_pricer

        return fine_tuned_pricer(OpenAI(), args.llm)
    if name == "qlora":
        from anvil.models.qlora import qlora_pricer

        return qlora_pricer(args.llm or config.FINETUNED_MODEL, None if args.llm else config.FINETUNED_REVISION)
    remote = None
    if name == "specialist" or (name == "ensemble" and config.USE_SPECIALIST):
        import modal

        remote = modal.Cls.from_name(config.MODAL_APP, "Anvil")()
        if name == "specialist":
            return lambda item: remote.price.remote(item.summary)
    from anvil import rag

    rag_predict = rag.rag_pricer(rag.get_collection(), rag.load_encoder(), args.llm or config.FRONTIER_MODEL, **kwargs)
    if name == "rag":
        return rag_predict
    from anvil.data.items import extract_price
    from anvil.data.parser import MAX_PRICE, MIN_PRICE
    from anvil.models.deep_neural_network import DeepNeuralNetworkPricer

    dnn = DeepNeuralNetworkPricer()
    dnn.load(args.weights)
    weights = config.active_ensemble_weights()

    def ensemble_predict(item) -> float:
        estimates = {"frontier": extract_price(rag_predict(item)), "neural_network": dnn(item)}
        if remote:
            estimates["specialist"] = remote.price.remote(item.summary)
        return sum(weights[name] * min(max(value, MIN_PRICE), MAX_PRICE) for name, value in estimates.items())

    return ensemble_predict


def cmd_evaluate(args) -> None:
    from anvil.data import store
    from anvil.evaluation import evaluate

    train, _, test = store.load_splits("items", args.lite)
    predictor = build_predictor(args, train)
    data = prompts_test(args.lite) if args.model == "qlora" else test
    workers = 1 if args.model in ("nn", "dnn", "qlora") else args.workers
    evaluate(predictor, data, title=args.title or args.llm or args.model, size=args.size, workers=workers, show=args.chart)


def cmd_train_dnn(args) -> None:
    from anvil.data import store
    from anvil.models.deep_neural_network import DeepNeuralNetworkPricer, DeepNeuralNetworkTrainer

    train, val, _ = store.load_splits("items", args.lite)
    dnn = DeepNeuralNetworkPricer()
    DeepNeuralNetworkTrainer(dnn, train, val[:1000]).train(args.epochs)
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    dnn.save(args.weights)
    print(f"Saved weights to {args.weights}")


def cmd_finetune_openai(args) -> None:
    from openai import OpenAI

    client = OpenAI()
    if args.job:
        job = client.fine_tuning.jobs.retrieve(args.job)
        print(f"{job.id} status={job.status} model={job.fine_tuned_model}")
        for event in client.fine_tuning.jobs.list_events(fine_tuning_job_id=args.job, limit=10).data:
            print(f"  {event.message}")
        return
    from anvil.data import store
    from anvil.models.finetune_openai import start_job

    train, val, _ = store.load_splits("items", args.lite)
    print("Started job", start_job(client, train[: args.train_size], val[: args.val_size], config.JSONL_DIR))


def cmd_finetune_qlora(args) -> None:
    from anvil.data import store
    from anvil.models.qlora import train

    folder = store.local_dir("items_prompts", args.lite)
    source = (
        {"train": folder / "train.jsonl", "val": folder / "validation.jsonl"}
        if (folder / "train.jsonl").exists()
        else config.dataset_name("items_prompts", args.lite)
    )
    print("Pushed adapter", train(source, config.require_hf_user(), epochs=args.epochs, report_to=args.report_to))


def cmd_run(args) -> None:
    from anvil.agents.framework import DealAgentFramework, reset_memory

    if args.reset_memory:
        reset_memory()
    for opportunity in DealAgentFramework(autonomous=args.autonomous).run():
        print(f"${opportunity.deal.price:>8.2f}  est ${opportunity.estimate:>8.2f}  {opportunity.deal.url}")


def cmd_app(args) -> None:
    from anvil.app import App

    App(autonomous=args.autonomous).launch(share=args.share)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="anvil", description="The Price is Right: product price prediction and deal-hunting agents")
    commands = root.add_subparsers(dest="command", required=True)

    def command(name: str, handler, help: str, lite: bool = True) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help)
        sub.set_defaults(handler=handler)
        if lite:
            sub.add_argument("--full", dest="lite", action="store_false", help="use the full dataset instead of lite")
        return sub

    sub = command("curate", cmd_curate, "load Amazon categories, dedupe, sample and split", lite=False)
    sub.add_argument("--categories", nargs="+", default=config.CATEGORIES)
    sub.add_argument("--sample-size", type=int, default=820_000)
    sub.add_argument("--workers", type=int, default=None)
    sub.add_argument("--per-category", type=int, help="randomly keep at most this many items per category, to save memory")
    sub.add_argument("--force", action="store_true", help="ignore data/raw_cache and redownload/reparse every category")
    sub.add_argument("--push", action="store_true")

    sub = command("summarize", cmd_summarize, "rewrite items into concise summaries with an LLM")
    sub.add_argument("mode", choices=["submit", "fetch", "local"])
    sub.add_argument("--model")
    sub.add_argument("--workers", type=int, default=4)
    sub.add_argument("--push", action="store_true")

    sub = command("prompts", cmd_prompts, "build prompt/completion pairs for fine-tuning")
    sub.add_argument("--cutoff", type=int, default=110)
    sub.add_argument("--push", action="store_true")

    sub = command("index", cmd_index, "embed training items into the Chroma vector store")
    sub.add_argument("--limit", type=int)

    sub = command("evaluate", cmd_evaluate, "evaluate a pricing model on the test set")
    sub.add_argument("model", choices=MODELS)
    sub.add_argument("--llm", help="LiteLLM model id, or fine-tuned model/adapter name")
    sub.add_argument("--reasoning-effort")
    sub.add_argument("--size", type=int, default=200)
    sub.add_argument("--workers", type=int, default=5)
    sub.add_argument("--epochs", type=int, default=2)
    sub.add_argument("--weights", type=Path, default=config.DNN_WEIGHTS)
    sub.add_argument("--title")
    sub.add_argument("--chart", action="store_true", help="open interactive charts")

    sub = command("train-dnn", cmd_train_dnn, "train the deep residual network")
    sub.add_argument("--epochs", type=int, default=5)
    sub.add_argument("--weights", type=Path, default=config.DNN_WEIGHTS)

    sub = command("finetune-openai", cmd_finetune_openai, "start or inspect an OpenAI fine-tuning job")
    sub.add_argument("--job", help="show status of an existing job instead of starting one")
    sub.add_argument("--train-size", type=int, default=100)
    sub.add_argument("--val-size", type=int, default=50)

    sub = command("finetune-qlora", cmd_finetune_qlora, "QLoRA fine-tune Llama 3.2 on a GPU")
    sub.add_argument("--epochs", type=int, default=1)
    sub.add_argument("--report-to", default="none", help="e.g. wandb")

    for name, handler, help in (("run", cmd_run, "run one pass of the deal-hunting agents"), ("app", cmd_app, "launch the Gradio UI")):
        sub = command(name, handler, help, lite=False)
        sub.add_argument("--autonomous", action="store_true", help="use the tool-calling planner")
        if name == "run":
            sub.add_argument("--reset-memory", action="store_true")
        else:
            sub.add_argument("--share", action="store_true")
    return root


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if getattr(args, "workers", 0) is None:
        from anvil.data.loaders import WORKERS

        args.workers = WORKERS
    args.handler(args)


if __name__ == "__main__":
    main()