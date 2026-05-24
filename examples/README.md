# Examples & reproduction

KeyStone is a drop-in, inference-time wrapper: at each round it draws **K**
candidate action chunks from a *shared* model context and selects one with the
geometry of the sampled chunks (the medoid of the largest cluster). This folder
shows how to run it — first standalone, then inside the two benchmark harnesses
used in the paper, then on any policy of your own.

## Start here: zero-dependency demo

No model or GPU required. Fakes a bimodal flow-matching policy and shows
KeyStone recovering the dominant mode as a *real* sampled chunk:

```bash
pip install -e ..            # install the keystone package
python minimal_quickstart.py
```

→ [`minimal_quickstart.py`](minimal_quickstart.py)

## Reproduce the paper

Each guide is **self-contained**: clone the fork at its `keystone` branch,
install KeyStone into that environment, and run. The selector itself is
maintained only here — both forks `import keystone`, so the numbers come from
this exact code.

| Guide | Models | Benchmark |
|---|---|---|
| [`README_lerobot.md`](README_lerobot.md) | SmolVLA, π0.5 | LIBERO |
| [`README_vla_eval.md`](README_vla_eval.md) | GR00T N1.6, X-VLA, StarVLA | SimplerEnv |

## Add KeyStone to your own policy

[`README_custom.md`](README_custom.md) is a framework-agnostic integration
guide. It covers the two patterns the forks use, each backed by a runnable
template you can copy:

- **Pattern A — override the sampler** ([`flow_matching_policy.py`](flow_matching_policy.py)):
  subclass a policy and replace its action-generation method. *(LeRobot style.)*
- **Pattern B — monkey-patch generate** ([`server_monkeypatch.py`](server_monkeypatch.py)):
  intercept a served model's `generate_actions` without touching the simulator.
  *(vla-eval style.)*
