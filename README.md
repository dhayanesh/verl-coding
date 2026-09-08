# Coding GRPO with verl

This is a small training project for `google/gemma-3-1b-it`. It uses:

- **verl** for GRPO training
- **vLLM**, launched and managed by verl, for generating four solutions per problem
- **Sandbox Fusion** in a separate Docker process for safely running generated Python
- pass rate across the problem's tests as the reward

The training loop is:

```text
coding prompt -> 4 vLLM solutions -> execute tests -> rewards -> GRPO update
```

## Files

```text
configs/coding.yaml       GRPO, vLLM, model, and batch settings
data/train/              43 GitHub-safe shards, 21,746 training problems
data/validation/          3 GitHub-safe shards, 928 validation problems
models/gemma-3-1b-it/     local Gemma checkpoint and tokenizer
reward.py                 extracts Python and scores it in Sandbox Fusion
scripts/start_sandbox.sh  starts the code-execution service
scripts/train.sh          starts training
requirements.txt          Python dependencies
```

Each Parquet shard is below 50 MiB, so the data can be stored in a normal GitHub
repository without Git LFS. The data is the stdin/stdout coding subset of
`PRIME-RL/Eurus-2-RL-Data` at
revision `9776b13264b5aaa0b16495fcf086a0a8d86fd655`. Exact validation prompt
matches were removed from training. Function-call tasks were excluded because
their test encodings are inconsistent with the stock Sandbox Fusion wrapper.

## Run

Install the environment if needed:

```bash
python -m pip install -r requirements.txt
```

Start Sandbox Fusion on a Docker-capable machine. It uses CPUs and does not need
a GPU:

```bash
SANDBOX_BIND=0.0.0.0 bash scripts/start_sandbox.sh
```

On the training machine, set its reachable URL and start training:

```bash
export SANDBOX_URL=http://SANDBOX_HOST:8080/run_code
bash scripts/train.sh
```

verl starts vLLM itself and synchronizes it with the updated model after each
GRPO step. Do not start a separate `vllm serve` process.

For a two-step pipeline check using small slices of the full files:

```bash
bash scripts/train.sh \
  data.train_max_samples=32 \
  data.val_max_samples=8 \
  trainer.total_training_steps=2 \
  trainer.test_freq=1 \
  trainer.save_freq=1
```

For one machine with eight GPUs:

```bash
bash scripts/train.sh \
  trainer.n_gpus_per_node=8 \
  data.train_batch_size=64 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  reward.num_workers=8 \
  ray_kwargs.ray_init.num_cpus=32
```

Checkpoints, rollouts, validation generations, and TensorBoard data are written
under `outputs/`. Override any setting by appending a Hydra argument to the
training command.

The reward is `passed_tests / total_tests`. A missing final `python` code block
receives zero. Wrong output, syntax errors, runtime errors, and timeouts fail the
affected tests. Sandbox service failures stop the run so infrastructure problems
are not recorded as model failures.
