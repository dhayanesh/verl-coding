"""A bounded experiment runner. Copilot chooses experiments; verl trains the model.

Run this inside the training pod. No Hugging Face downloads are needed.
"""

import argparse
import collections
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "outputs/autoresearch"
FIELDS = ["name", "split", "status", "lr", "kl", "seed", "accuracy",
          "mean_reward", "initial_accuracy", "problems", "seconds"]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def initialize():
    """Freeze configuration and split validation before any experiment is selected."""
    model = Path(os.environ["MODEL_PATH"]).resolve()
    sandbox = os.environ["SANDBOX_URL"]
    if not model.is_dir() or not sandbox:
        raise ValueError("Set a local MODEL_PATH and a reachable SANDBOX_URL")
    os.environ.update(PROJECT_DIR=str(ROOT), MODEL_PATH=str(model), SANDBOX_URL=sandbox)
    settings = OmegaConf.to_container(OmegaConf.load(ROOT / "autoresearch.yaml"), resolve=True)
    for key in ["gpus", "nodes", "steps", "tuning_prompts", "train_batch_size",
                "ppo_mini_batch_size", "timeout_seconds", "max_tuning_runs", "max_final_runs"]:
        if settings[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if STUDY.exists():
        raise ValueError(f"Study already exists at {STUDY}; use it or a fresh checkout")

    config = OmegaConf.load(ROOT / "configs/coding.yaml")
    config.trainer.experiment_name = "autoresearch"
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    config.trainer.nnodes = settings["nodes"]
    config.trainer.n_gpus_per_node = settings["gpus"]
    config.trainer.total_training_steps = settings["steps"]
    # Enough epochs even if prompt filtering makes the dataset small.
    config.trainer.total_epochs = settings["steps"]
    config.trainer.resume_mode = "disable"
    config.trainer.test_freq = settings["steps"]
    config.trainer.save_freq = settings["steps"]
    config.trainer.max_actor_ckpt_to_keep = 1
    config.trainer.logger = ["console"]
    config.trainer.rollout_data_dir = None
    config.data.train_batch_size = settings["train_batch_size"]
    config.actor_rollout_ref.actor.ppo_mini_batch_size = settings["ppo_mini_batch_size"]
    config.reward.num_workers = settings["reward_workers"]
    config.ray_kwargs.ray_init = (
        {"address": settings["ray_address"]} if settings["ray_address"]
        else {"address": "local", "num_cpus": settings["local_cpus"]}
    )
    # Explicitly propagate these paths to workers on an existing Ray cluster.
    config.ray_kwargs.ray_init.runtime_env = {"env_vars": {
        "PROJECT_DIR": str(ROOT), "MODEL_PATH": str(model), "SANDBOX_URL": sandbox,
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }}

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    tables = [pq.read_table(path) for path in config.data.val_files]
    validation = pa.concat_tables(tables)
    groups = collections.defaultdict(list)
    for i, prompt in enumerate(validation["prompt"].to_pylist()):
        tokens = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        if len(tokens) <= config.data.max_prompt_length:
            key = json.dumps(prompt, sort_keys=True)
            groups[key].append(i)
    keys = sorted(groups)
    random.Random(settings["seed"]).shuffle(keys)
    n = settings["tuning_prompts"]
    if not 0 < n < len(keys):
        raise ValueError(f"tuning_prompts must be below {len(keys)} eligible unique prompts")
    # Exact duplicate prompts always stay in the same split.
    split_indices = {
        "tune": sorted(i for key in keys[:n] for i in groups[key]),
        "final": sorted(i for key in keys[n:] for i in groups[key]),
    }
    STUDY.mkdir(parents=True)
    for split, indices in split_indices.items():
        pq.write_table(validation.take(indices), STUDY / f"{split}.parquet", compression="zstd")
    config.data.val_files = [str(STUDY / "tune.parquet")]
    OmegaConf.save(config, STUDY / "base.yaml")
    tracked = [ROOT / "reward.py", ROOT / "scripts/autoresearch.py", STUDY / "base.yaml",
               STUDY / "tune.parquet", STUDY / "final.parquet"]
    tracked += [Path(p) for p in config.data.train_files]
    tracked += sorted(p for p in model.iterdir() if p.is_file())
    print("Fingerprinting fixed code, data, and model...", flush=True)
    manifest = {"settings": settings, "rows": {k: len(v) for k, v in split_indices.items()},
                "hashes": {str(p): digest(p) for p in tracked}}
    write_json(STUDY / "study.json", manifest)
    with (STUDY / "results.tsv").open("w") as stream:
        csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t").writeheader()
    print(json.dumps(manifest["rows"]), "Study initialized.")


def evaluate(path, expected):
    """Read verl's validation generations; never infer success from missing logs."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} validation rows, found {len(rows)}")
    scores = [float(row["score"]) for row in rows]
    if any(not 0 <= score <= 1 for score in scores):
        raise ValueError("Invalid validation score")
    return {"accuracy": sum(score == 1.0 for score in scores) / len(scores),
            "mean_reward": sum(scores) / len(scores), "problems": len(scores)}, rows


def run(args):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        raise ValueError("Use only letters, digits, underscores, and hyphens for run names")
    if not 0 < args.lr <= 1e-4 or not 0 <= args.kl <= 1:
        raise ValueError("Use 0 < lr <= 1e-4 and 0 <= kl <= 1")
    manifest = json.loads((STUDY / "study.json").read_text())
    settings = manifest["settings"]
    split = "final" if args.final else "tune"
    # One live run per study, including through separate kubectl exec sessions.
    with (STUDY / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        history = list((STUDY / "runs").glob("*/result.json"))
        previous = [json.loads(p.read_text()) for p in history]
        if any(row["status"] == "running" for row in previous):
            raise ValueError("An earlier run has no final status. Inspect its process and Ray resources first.")
        limit = settings["max_final_runs" if args.final else "max_tuning_runs"]
        if sum(row["split"] == split for row in previous) >= limit:
            raise ValueError(f"{split} run budget exhausted")
        baseline_ok = any(row["status"] == "ok" and row["split"] == "tune"
                          and row["lr"] == 1e-6 and row["kl"] == 0.001 for row in previous)
        if not baseline_ok and (args.final or args.lr != 1e-6 or args.kl != 0.001):
            raise ValueError("Complete a successful tuning baseline with lr=1e-6 and kl=0.001 first")
        for name, expected in manifest["hashes"].items():
            if digest(Path(name)) != expected:
                raise ValueError(f"Fixed study input changed: {name}")
        directory = STUDY / "runs" / args.name
        directory.mkdir(parents=True, exist_ok=False)
        seed = settings["seed"] if args.seed is None else args.seed
        result = {"name": args.name, "split": split, "status": "running", "lr": args.lr,
                  "kl": args.kl, "seed": seed, "pid": os.getpid(), "hypothesis": args.hypothesis}
        result["versions"] = {name: importlib.metadata.version(name)
                              for name in ["verl", "vllm", "torch", "transformers", "ray"]}
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True)
        result["git_commit"] = revision.stdout.strip() or "unavailable"
        write_json(directory / "result.json", result)
        config = OmegaConf.load(STUDY / "base.yaml")
        config.actor_rollout_ref.actor.optim.lr = args.lr
        config.actor_rollout_ref.actor.kl_loss_coef = args.kl
        config.data.seed = seed
        config.actor_rollout_ref.actor.data_loader_seed = seed
        config.actor_rollout_ref.actor.fsdp_config.seed = seed
        config.actor_rollout_ref.ref.fsdp_config.seed = seed
        config.actor_rollout_ref.rollout.engine_kwargs = {"vllm": {"seed": seed}}
        config.data.val_files = [str(STUDY / f"{split}.parquet")]
        config.trainer.experiment_name = args.name
        config.trainer.default_local_dir = str(directory / "checkpoints")
        config.trainer.validation_data_dir = str(directory / "validation")
        config.hydra.run = {"dir": str(directory / "hydra")}
        OmegaConf.save(config, directory / "experiment.yaml")
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        command = [sys.executable, "-m", "verl.trainer.main_ppo", "--config-path",
                   str(directory), "--config-name", "experiment"]
        started = time.monotonic()
        child = None
        try:
            with (directory / "resolved.yaml").open("w") as resolved:
                subprocess.run(command + ["--cfg", "job", "--resolve"], cwd=ROOT, env=env,
                               stdout=resolved, stderr=subprocess.PIPE, check=True, timeout=120)
            with (directory / "train.log").open("w") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                result["training_pid"] = child.pid
                write_json(directory / "result.json", result)
                result["exit_code"] = child.wait(timeout=settings["timeout_seconds"])
            if result["exit_code"] != 0:
                raise RuntimeError(f"Training exited with code {result['exit_code']}")
            initial, initial_rows = evaluate(directory / "validation/0.jsonl", manifest["rows"][split])
            final, final_rows = evaluate(directory / f"validation/{settings['steps']}.jsonl",
                                         manifest["rows"][split])
            if collections.Counter(r["input"] for r in initial_rows) != collections.Counter(r["input"] for r in final_rows):
                raise ValueError("Initial and final evaluation prompts differ")
            result.update(final, initial_accuracy=initial["accuracy"], status="ok")
        except (Exception, KeyboardInterrupt) as error:
            result.update(status="failed", error=str(error) or "Interrupted")
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            result["seconds"] = round(time.monotonic() - started, 1)
            write_json(directory / "result.json", result)
            with (STUDY / "results.tsv").open("a") as stream:
                csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore").writerow(result)
        print(json.dumps(result, indent=2))
        if result["status"] != "ok":
            raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    commands.add_parser("status")
    experiment = commands.add_parser("run")
    experiment.add_argument("name")
    experiment.add_argument("--lr", type=float, default=1e-6)
    experiment.add_argument("--kl", type=float, default=0.001)
    experiment.add_argument("--seed", type=int)
    experiment.add_argument("--hypothesis", default="Baseline configuration")
    experiment.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        initialize()
    elif args.command == "run":
        run(args)
    else:
        for path in sorted((STUDY / "runs").glob("*/result.json")):
            print(path.parent.name, path.read_text())
