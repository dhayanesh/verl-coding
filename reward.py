"""The only task-specific part of GRPO: execute code and return its test pass rate.

Generated code runs in Sandbox Fusion, never in the trainer's Python process.
We reuse verl's stdin/stdout test executor and its existing comparison rules.
Read compute_score from top to bottom first.
"""

import json
import os
import re
import threading

from verl.utils.reward_score.sandbox_fusion.utils import check_correctness

# This limit is per reward worker, not a cluster-wide limit.
SANDBOX_SLOTS = threading.Semaphore(4)


def extract_code(response):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", response, flags=re.DOTALL)
    return blocks[-1].strip() if blocks else None


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  sandbox_url=None, timeout=10, **kwargs):
    """verl calls this once per generated solution. All metrics are per solution."""
    tests = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if not isinstance(tests, dict) or tests.get("fn_name"):
        raise ValueError("Use the prepared stdin/stdout dataset")
    inputs, outputs = tests.get("inputs"), tests.get("outputs")
    if not isinstance(inputs, list) or not inputs or not isinstance(outputs, list):
        raise ValueError("Reward needs nonempty input and output test lists")
    if len(inputs) != len(outputs) or any(output is None for output in outputs):
        raise ValueError("Reward test inputs/outputs are invalid")
    if not all(isinstance(value, str) for value in inputs + outputs):
        raise ValueError("Prepared tests must contain stdin/stdout strings")

    code = extract_code(solution_str)
    if not code:
        return {"score": 0.0, "acc": 0.0, "format_error": 1.0,
                "runtime_error_fraction": 0.0, "timeout_fraction": 0.0}

    url = sandbox_url or os.environ.get("SANDBOX_URL", "http://127.0.0.1:8080/run_code")
    results = []
    # Tests of one solution run sequentially; different solutions can run concurrently.
    # This avoids the helper creating hundreds of threads for a long test list.
    for stdin, expected in zip(inputs, outputs, strict=True):
        case_results, metadata = check_correctness(
            sandbox_fusion_url=url,
            in_outs={"inputs": [stdin], "outputs": [expected]},
            generation=code,
            timeout=timeout,
            memory_limit_mb=1024,
            language="python",
            concurrent_semaphore=SANDBOX_SLOTS,
        )
        # -1 means infrastructure failure. Do not teach the model that this was wrong code.
        if len(case_results) != 1 or case_results[0] == -1:
            failures = [m.get("status", "unknown") for m in metadata if m]
            raise RuntimeError(f"Sandbox infrastructure failure: {failures}")
        results.append(case_results[0])

    passed = sum(result is True for result in results)  # Negative error codes are truthy!
    total = len(results)
    return {
        "score": passed / total,             # GRPO optimizes fractional test pass rate.
        "acc": float(passed == total),      # Evaluation also reports complete solutions.
        "format_error": 0.0,
        "runtime_error_fraction": sum(r in (-2, -4) for r in results) / total,
        "timeout_fraction": sum(r == -3 for r in results) / total,
    }
