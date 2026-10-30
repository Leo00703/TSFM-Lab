# Security policy

TSFM-Lab is a **local** tool: it runs on your own machine, listens on `127.0.0.1` by default and sends nothing anywhere except the downloads you
ask for (model weights and public datasets from Hugging Face, pinned to a commit and checked against a SHA-256). The page
[Security model](docs/SECURITY-MODEL.md) says what it protects against and, as plainly, what it does not (for example, there is no login when you use
`--tailscale`).

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Use GitHub's private reporting instead: the repository's **Security** tab, then
**Report a vulnerability**. Say what you found, how to reproduce it and which version (`git rev-parse --short HEAD`) you ran. You will get an
answer as soon as the maintainer reads it; this is a one-person project, so there is no response-time guarantee.

## What is in scope

- Anything that lets a web page, another local user or a file the app reads run code, read files outside the project or reach the server from
  another host (the `Host`/`Origin` checks, the upload and download paths, the run and dataset identifiers taken from URLs).
- A weight or dataset file that gets used without being checked (the `safetensors`-only rule, the pinned commits and checksums).

## Dependencies

GitHub's Dependabot alerts are on, and the pinned versions in `uv.lock` are also checked against the OSV database with `pip-audit`
(last on 30 October 2026, for the CPU, CUDA and macOS configurations): **no known vulnerability is open.** On that date the lock file moved
every platform to PyTorch 2.13.0 and `setuptools` 84.2.1, which closed GHSA-rrmf-rvhw-rf47 (torch, `torch.jit.script` memory corruption, fixed in
2.13.0) and GHSA-xmj3-9gf7-h52w, GHSA-h35f-9h28-mq5c and PYSEC-2026-3447 (`setuptools`, fixed in 83 and 84.1). The app used none of the affected
parts. `pip-audit` cannot look up the local builds `torch 2.13.0+cpu` and `+cu130` (they are not on PyPI); it checks the same release on the
macOS configuration and Dependabot reports all alerts as fixed.

The OSV audit is not run automatically: repeat it after updating the lock file
(`uv export --no-hashes -o requirements.txt`, then `uvx pip-audit -r requirements.txt --no-deps`).

Out of scope: vulnerabilities in the libraries the app depends on (report them upstream; `uv.lock` pins the tested versions), and the models'
own behaviour (what they forecast, their licences, their training data).
