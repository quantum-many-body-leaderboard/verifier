"""Save and load a JAX parameter tree as .npz without flax (numpy only).

    from jax_npz import save_tree
    save_tree("state.npz", params)      # nested dicts of arrays; keys joined with "/"

The verifier reads the same layout back for `--state jax:file.py:NAME --params state.npz`
(qrec_verify/external.py: load_npz_tree). Lists and tuples become dicts keyed "0", "1", ...;
other trees (Equinox modules) are saved by their own code and loaded by a `load_params(path)`
in the state file.
"""

import numpy as np


def flatten(tree, prefix=""):
    if isinstance(tree, dict):
        items = tree.items()
    elif isinstance(tree, (list, tuple)):
        items = ((str(i), v) for i, v in enumerate(tree))
    else:
        return {prefix: np.asarray(tree)}
    out = {}
    for k, v in items:
        k = str(k)
        if "/" in k:
            raise ValueError(f"key {k!r} contains '/'")
        out.update(flatten(v, f"{prefix}/{k}" if prefix else k))
    return out


def save_tree(path, tree):
    np.savez(path, **flatten(tree))


def load_tree(path):
    tree = {}
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            node = tree
            parts = key.split("/")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = np.asarray(z[key])
    return tree
