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

GitHub's Dependabot alerts are on, and the pinned versions in `uv.lock` were also checked against the OSV database with `pip-audit` on
29 October 2026. Open advisories, none of which the app can reach:

- `setuptools` 81.0.0 (GHSA-xmj3-9gf7-h52w, GHSA-h35f-9h28-mq5c and PYSEC-2026-3447, fixed in 83 and 84.1). PyTorch 2.11 requires
  `setuptools<82`, and the macOS build is still on 2.11 (PyTorch 2.13 has no such limit), so the lock file cannot move it yet. The advisories
  concern converting an untrusted legacy wheel on Windows and building a source distribution on macOS, which the app never does (it does not
  call `setuptools` at all; it is only a dependency of PyTorch).
- `torch` 2.11.0 (GHSA-rrmf-rvhw-rf47, low, fixed in 2.13.0): memory corruption through `torch.jit.script`. The app never calls it. Windows and
  Linux (CPU and CUDA 13.0) are on 2.13.0 since 30 October 2026; macOS stays on 2.11.0 until the Apple GPU path has been re-checked with 2.13 on a Mac.
  The alert stays open for as long as `uv.lock` holds the macOS build.

The OSV audit is not run automatically: repeat it after updating the lock file
(`uv export --no-hashes -o requirements.txt`, then `uvx pip-audit -r requirements.txt --no-deps`).

Out of scope: vulnerabilities in the libraries the app depends on (report them upstream; `uv.lock` pins the tested versions), and the models'
own behaviour (what they forecast, their licences, their training data).
