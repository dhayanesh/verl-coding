# Copilot autoresearch for coding GRPO

You are operating from VS Code Agent mode. Execute training commands inside the
user's Kubernetes Ray pod through the VS Code terminal. This is an adaptation of
https://github.com/karpathy/autoresearch to verl, not its original nanochat task.

## Setup once

1. Obtain the Kubernetes context, namespace, pod, container, project path,
   MODEL_PATH, SANDBOX_URL, and assigned GPU count from the user/environment.
   Keep these values in the session, not in public source files.
2. Inspect the pod and Ray cluster read-only. Confirm which GPU nodes are assigned
   to this job. `ray_address: auto` attaches to an existing Ray cluster; it does not
   pin training to the pod entered with kubectl. The GPU allocation must already
   provide suitable isolation. Do not restart Ray or terminate unrelated jobs.
3. Ensure the project, dataset, model, and outputs are on persistent storage at
   identical absolute paths on every Ray node that can run this job. All Ray
   workers need the installed dependencies and access to Sandbox Fusion.
4. Ensure the code/config edited in VS Code is copied to that project directory
   before initialization. A local editor save does not update a separate pod.
5. Edit autoresearch.yaml for the assigned hardware. For one node with eight GPUs,
   start with gpus=8, nodes=1, train_batch_size=64,
   ppo_mini_batch_size=64, reward_workers=8. These are unvalidated starting values.
6. Export MODEL_PATH (the original local Gemma checkpoint) and SANDBOX_URL inside
   the pod. Execute `python scripts/autoresearch.py init` there. It freezes config,
   fingerprints code/data/model, and creates a tuning split and a reserved final
   split from validation. Overlong validation prompts are excluded before splitting.
   Exact duplicate prompts remain together. It does not download anything from HF.

The study is outputs/autoresearch/. Initialization can happen only once. All
following commands run inside the pod's project directory. Do not change frozen
inputs or study.json. For a different study protocol, use a separate checkout and
output directory. Do not reset or delete an existing study to bypass its budget.

## Baseline first

Run `python scripts/autoresearch.py run baseline` before trying any candidate.
Use a detached process so the kubectl exec connection can close. For example,
inside the pod's shell, after changing into the project directory:

```bash
nohup python -u scripts/autoresearch.py run baseline \
  > outputs/autoresearch/launch-baseline.log 2>&1 < /dev/null &
```

Use kubectl exec with a shell to execute that command in the remote directory.
Do not rely on a background process in the local VS Code terminal to survive a
disconnection. nohup survives terminal disconnection, not pod eviction; persistent
storage preserves results but does not resume a killed process automatically.

Monitor `python scripts/autoresearch.py status`, the launch log, and
outputs/autoresearch/runs/baseline/train.log. Each run records its own driver PID,
training PID, effective config, resolved verl config, versions, and input hashes
(the latter are in the study manifest).

Default budget is 20 optimizer iterations with fixed batch and rollout count,
and a 2-hour timeout. This is a pilot, not proof of useful learning. Inspect runtime
and reward variation before choosing a longer budget for a separate study. Never
change the budget midway through a comparison.

## Research loop

- Run one experiment at a time. Never mutate inputs while a run is active.
- Tune ONLY --lr and --kl. Do not edit reward.py, datasets, the model, evaluation,
  the runner, frozen base.yaml, or installed packages to improve a measured score.
- Start every run from the original model; the runner disables checkpoint resume.
  Keep a promising configuration, not extra training from its last checkpoint.
- Write one hypothesis with --hypothesis and use a unique run name.
- First consider lr=5e-7 or 2e-6 versus the baseline 1e-6, keeping kl=0.001.
  Then consider kl=0.0005 or 0.002 at the best learning rate. These are search
  candidates, not claims of optimal values.
- Reserve some of the six tuning slots for repeating baseline and the leading
  candidate with the same alternate seed using --seed. Failed runs consume slots.
- Compare only successful runs with the same protocol. Primary metric is accuracy:
  the fraction of problems passing ALL stored tests. Mean fractional test reward
  is secondary. Never choose configurations based only on training reward.
- Report a tie as a tie. At 64 problems one extra solved problem is about 1.6
  percentage points; small gains need confirmation. RL and GPU execution remain
  stochastic even with recorded seeds.
- If all groups get identical rewards, report lack of learning signal. Do not
  weaken tests, shorten the evaluation set, or invent successful measurements.

Example candidate (launch with nohup as above):

```bash
python scripts/autoresearch.py run lower-lr --lr 5e-7 --kl 0.001 \
  --hypothesis "A smaller learning rate may improve stability"
```

The runner records results in outputs/autoresearch/results.tsv and per-run
result.json. Failed runs have no fabricated accuracy. A missing final validation
file or changed validation count makes a run fail. Keep all logs and checkpoints.
Do not modify main, auto-push experiments, or use git reset --hard.

## Failures and stopping

The runner stops its own local training process group on timeout. Ray workers are
remote processes; after failure, inspect Ray to verify that this job has released
its resources before starting another run. Never use `ray stop` on a shared cluster.
If a run remains marked running after pod death, inspect its recorded PIDs and Ray
job state; do not silently treat it as successful or launch a replacement over it.
Stop after two consecutive infrastructure failures and report the exact blocker.
The runner's lock covers this study only, not independent projects on the cluster.

Stop when the configured tuning budget is exhausted. Select the winner using the
tuning results BEFORE reading the reserved final data. Then run at most two final
experiments, baseline and the selected configuration, with matching seeds:

```bash
python scripts/autoresearch.py run final-baseline --final
python scripts/autoresearch.py run final-selected --final --lr <selected-lr> --kl <selected-kl>
```

These commands each train afresh and evaluate on the reserved split. Do not use
those final results to launch another search on the same study. Summarize the
baseline, candidates, failures, confirmed winner or lack of improvement, runtime,
and final evaluation. These are local Eurus-subset results, not a paper reproduction.
