from __future__ import annotations

from exp7.runners import agentbench, agentdojo, appworld, bfcl, injecagent

RUNNERS = {
    "agentdojo": agentdojo.run,
    "injecagent": injecagent.run,
    "bfcl": bfcl.run,
    "agentbench": agentbench.run,
    "appworld": appworld.run,
}


def get_runner(name: str):
    if name not in RUNNERS:
        raise KeyError(f"Unknown benchmark: {name}. Choose from {list(RUNNERS)}")
    return RUNNERS[name]
