"""Repo-local Kitaru runtime probes for thin real-runtime durability tests.

Each scenario is intentionally small:
- ``replay`` proves replay from a completed checkpoint boundary with overrides.
- ``wait`` proves wait -> pending_wait -> input -> resume on a real local stack.

The module lives under ``tests/`` so ZenML/Kitaru can resolve it as source code
inside the repository root during pipeline compilation.
"""

import json
import sys

from kitaru import checkpoint, configure, create_stack, flow, wait
from kitaru.client import KitaruClient


def _configure_local_stack() -> None:
    """Initialize an isolated local Kitaru/ZenML stack for the probe run."""
    configure()
    create_stack("test-local", activate=True)


_REPLAY_CALLS = {"first": 0, "second": 0, "third": 0}


@checkpoint
def replay_first(x: int) -> int:
    _REPLAY_CALLS["first"] += 1
    return x + 1


@checkpoint
def replay_second(y: int) -> int:
    _REPLAY_CALLS["second"] += 1
    return y * 2


@checkpoint
def replay_third(z: int) -> int:
    _REPLAY_CALLS["third"] += 1
    return z + 3


@flow
def replay_probe_flow(x: int) -> int:
    a = replay_first(x)
    b = replay_second(a)
    return replay_third(b)


_WAIT_CALLS = {"produce": 0, "finalize": 0}


@checkpoint
def wait_produce() -> str:
    _WAIT_CALLS["produce"] += 1
    return "draft"


@checkpoint
def wait_finalize(approved: bool, draft: str) -> str:
    _WAIT_CALLS["finalize"] += 1
    return f"{draft}:{approved}"


@flow
def wait_probe_flow() -> str:
    draft = wait_produce()
    approved = wait(
        name="approve_draft",
        question="Approve?",
        schema=bool,
        timeout=1,
    )
    return wait_finalize(approved, draft)


def run_replay_probe() -> dict[str, object]:
    _configure_local_stack()

    handle = replay_probe_flow.run(1)
    first_result = handle.wait()

    replayed = replay_probe_flow.replay(
        handle.exec_id,
        from_="replay_second",
        overrides={"checkpoint.replay_second": 100},
        x=1,
    )
    replay_result = replayed.wait()

    return {
        "scenario": "replay",
        "initial_result": first_result,
        "replay_result": replay_result,
        "calls": dict(_REPLAY_CALLS),
    }


def run_wait_probe() -> dict[str, object]:
    _configure_local_stack()

    handle = wait_probe_flow.run()
    client = KitaruClient()
    waits = client.executions.pending_waits(handle.exec_id)
    client.executions.input(handle.exec_id, wait="approve_draft", value=True)
    resumed = client.executions.resume(handle.exec_id)
    result = handle.wait()

    return {
        "scenario": "wait",
        "result": result,
        "wait_names": [pending_wait.name for pending_wait in waits],
        "resume_status": str(getattr(resumed, "status", "")),
        "calls": dict(_WAIT_CALLS),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in {"replay", "wait"}:
        print("usage: python tests/_runtime_kitaru_probe.py [replay|wait]", file=sys.stderr)
        return 2

    if argv[1] == "replay":
        payload = run_replay_probe()
    else:
        payload = run_wait_probe()

    print("JSON_RESULT=" + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
