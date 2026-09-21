# Environments

Base execution environments (where agent commands run):

* `kubernetes.py` - Execute code in a Kubernetes pod (primary, used for batch rollouts)
* `docker.py` - Execute code in a docker or podman container
* `cube.py` - Execute code in a CubeSandbox micro-VM (`uv sync --extra cube`)
* `modal.py` - Execute code in a Modal Sandbox (`uv sync --extra modal`)
* `detached.py` - `DetachedExecMixin`: launch-and-poll `execute_detached` shared by the Kubernetes and Modal backends
* `local.py` - Execute code with `subprocess.run`

## Dataset environments

`datasets/` wraps a base environment with dataset-specific setup and reward
calculation (DeepSWE, opensource-code, ARVO, ...). See
`datasets/__init__.py` for the registry and routing rules, and
`rubric_judge.py` for the rubric-based reward stage that can stack on any dataset.
