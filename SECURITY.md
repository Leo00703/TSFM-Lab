# Security policy

TSFM-Lab is a **local** tool: it runs on your own machine, listens on `127.0.0.1` by default and sends nothing anywhere except the downloads you
ask for (model weights and public datasets from Hugging Face, pinned to a commit and checked against a SHA-256). The README section
[Security](README.md#security) says what it protects against and, as plainly, what it does not (for example, there is no login when you use
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

The pinned versions in `uv.lock` were checked against the OSV database with `pip-audit` on 29 October 2026. One package has known advisories:
`setuptools` 81.0.0 (PYSEC-2026-3447 and GHSA-xmj3-9gf7-h52w, fixed in 83 and 84.1). PyTorch 2.11 requires `setuptools<82`, so it cannot be moved
yet. The advisories concern converting an untrusted legacy wheel at install time on Windows, which the app never does (it does not call
`setuptools` at all; it is only a dependency of PyTorch). The audit is not run automatically: repeat it after updating the lock file
(`uv export --no-hashes -o requirements.txt`, then `uvx pip-audit -r requirements.txt --no-deps`).

Out of scope: vulnerabilities in the libraries the app depends on (report them upstream; `uv.lock` pins the tested versions), and the models'
own behaviour (what they forecast, their licences, their training data).
