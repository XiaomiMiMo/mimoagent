# Security Policy

## Reporting a vulnerability

Please do not open public issues for security problems. Report them through
GitHub's private vulnerability reporting on this repository
(Security → Report a vulnerability), or email the maintainers at the address
listed on the GitHub organization page. We aim to acknowledge reports within
five working days.

## Scope

mimoagent executes model-generated commands inside task environments (Docker
containers, Kubernetes pods, CubeSandbox micro-VMs). Treat those environments
as untrusted: give them only the network access, credentials and mounts the
task needs, and keep the controller host separate from them. Reports about
sandbox escapes, credential exposure through logs or trajectories, and unsafe
defaults are especially welcome.
