"""Command line: ``tsfm-lab <command>`` (or ``python -m tsfm_lab``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tsfm_lab.data import synthetic
from tsfm_lab.data.local import ColumnMapping, guess_mapping, load_file, read_table
from tsfm_lab.eval.backtest import BacktestConfig
from tsfm_lab.eval.runner import DEFAULT_BASELINES, run_experiment
from tsfm_lab.models.registry import default_registry
from tsfm_lab.store.store import RunStore
from tsfm_lab.system.devices import list_devices


def _table(rows: list[list[str]], header: list[str]) -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    for i, row in enumerate([header, *rows]):
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
        if i == 0:
            print("  ".join("-" * w for w in widths))


def cmd_models(_: argparse.Namespace) -> None:
    rows = []
    for i in default_registry().infos():
        c = i.capabilities
        rows.append([i.id, i.kind, "-" if i.params is None else f"{i.params / 1e6:.1f}M",
                     "yes" if c.known_covariates else "no", "-" if c.max_context is None else str(c.max_context),
                     ",".join(c.device_kinds), i.license.split(" (")[0]])
    _table(rows, ["id", "kind", "params", "covariates", "max_context", "devices", "license"])


def cmd_devices(_: argparse.Namespace) -> None:
    rows = [[d.id, d.name, "-" if d.memory_gb is None else f"{d.memory_gb} GB",
             "ok" if d.supported else f"UNSUPPORTED: {d.reason}"] for d in list_devices()]
    _table(rows, ["device", "name", "memory", "status"])


def cmd_datasets(_: argparse.Namespace) -> None:
    rows = [[s.id, ",".join(s.tags), s.description[:90]] for s in synthetic.SCENARIOS.values()]
    _table(rows, ["scenario", "tags", "description"])


def cmd_inspect(args: argparse.Namespace) -> None:
    frame = read_table(args.path)
    mapping = guess_mapping(frame)
    print(f"Guessed mapping: {mapping}")
    _, report = load_file(args.path, mapping)
    print(json.dumps(report, indent=1, default=str))


def _dataset(args: argparse.Namespace):
    if args.dataset.startswith("file:"):
        path = args.dataset[5:]
        frame = read_table(path)
        mapping = guess_mapping(frame)
        if args.targets:
            mapping = ColumnMapping(mapping.timestamp, args.targets.split(","), mapping.id)
        ds, report = load_file(path, mapping)
        for w in report["warnings"]:
            print(f"warning: {w}")
        return ds
    return synthetic.build(args.dataset)


def cmd_run(args: argparse.Namespace) -> None:
    ds = _dataset(args)
    cfg = BacktestConfig(horizon=args.horizon, n_windows=args.windows, max_series=args.max_series,
                         use_covariates=not args.no_covariates, max_context=args.max_context,
                         options={"symmetric_averaging": not args.no_symmetric,
                                  "mask_flat_covariates": args.mask_flat_covariates})
    baselines = tuple(b for b in args.baselines.split(",") if b)
    store = None if args.no_store else RunStore()
    print(f"dataset {ds.id}: {len(ds.series)} series, freq={ds.freq}, period={ds.seasonal_period}")
    result, run_id = run_experiment(args.model, ds, cfg, args.device, baselines, store, label=args.label,
                                    progress=lambda d, t, m: print(f"\r  {m}: {d}/{t}", end="", flush=True))
    print()
    rows = []
    for mid, o in result.models.items():
        s = o.summary
        rows.append([mid, o.device, f"{s['mase']['mean']:.3f}", f"{s['sql']['mean']:.3f}",
                     f"{s['coverage_80']['mean']:.2f}", f"{s['coverage_90']['mean']:.2f}",
                     f"{sum(o.batch_ms):.0f}", "-" if o.warmup_ms is None else f"{o.warmup_ms:.0f}"])
    _table(rows, ["model", "device", "MASE", "SQL", "cov80", "cov90", "warm ms", "cold ms"])
    main_model = result.models[args.model]
    for other, sk in main_model.skill.items():
        print(f"skill of {args.model} vs {other}: MASE {sk['mase']['skill']:+.1%}, SQL {sk['sql']['skill']:+.1%} "
              f"(n={sk['sql']['n']})")
    for mid, o in result.models.items():
        for w in o.warnings:
            print(f"warning [{mid}]: {w}")
        for n in o.notes:
            print(f"note [{mid}]: {n}")
    if run_id:
        print(f"saved run {run_id}")


def _tailscale() -> tuple[str, list[str]]:
    """(tailnet IPv4, host names to accept) from the local Tailscale client."""
    import shutil
    import subprocess

    exe = shutil.which("tailscale") or str(Path("C:/Program Files/Tailscale/tailscale.exe"))
    try:
        ip = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=10, check=False).stdout.split()[0]
        status = json.loads(subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=10, check=False).stdout)
    except (OSError, IndexError, ValueError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Tailscale is not available or not connected: {error}") from error
    dns = status["Self"]["DNSName"].rstrip(".")
    return ip, [ip, dns, dns.split(".")[0]]


def cmd_serve(args: argparse.Namespace) -> None:
    import socket
    import threading
    import webbrowser

    import uvicorn

    from tsfm_lab.api.app import create_app
    from tsfm_lab.api.build_css import ensure_css

    ensure_css()
    hosts = ["127.0.0.1", "localhost", "[::1]"]
    binds = ["127.0.0.1"]
    urls = [f"http://127.0.0.1:{args.port}"]
    if args.tailscale:
        ip, names = _tailscale()
        binds.append(ip)
        hosts += names
        urls += [f"http://{n}:{args.port}" for n in names[:2]]
    sockets = []
    for host in binds:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, args.port))
        sock.set_inheritable(True)
        sockets.append(sock)
    print("tsfm-lab reachable at:")
    for url in urls:
        print("  " + url)
    print("Listening only on " + ", ".join(binds) + (" (your tailnet; no authentication: anyone on it can use the app)" if args.tailscale else " (this machine only)")
          + ". Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(urls[0])).start()
    server = uvicorn.Server(uvicorn.Config(create_app(allowed_hosts=tuple(hosts)), log_level="warning"))
    server.run(sockets=sockets)


def cmd_runs(_: argparse.Namespace) -> None:
    rows = [[r["id"], r["dataset_id"], ",".join(r["models"]), ",".join(r["devices"]), str(r["horizon"]),
             str(r["n_items"])] for r in RunStore().list_runs()]
    _table(rows, ["run", "dataset", "models", "devices", "horizon", "items"]) if rows else print("no runs yet")


def cmd_show(args: argparse.Namespace) -> None:
    print(json.dumps(RunStore().meta(args.run_id), indent=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsfm-lab", description="Local sandbox for time-series foundation models")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="list registered models").set_defaults(fn=cmd_models)
    sub.add_parser("devices", help="list compute devices").set_defaults(fn=cmd_devices)
    sub.add_parser("datasets", help="list bundled synthetic scenarios").set_defaults(fn=cmd_datasets)
    p = sub.add_parser("inspect", help="validate a CSV/Parquet/Excel file"); p.add_argument("path"); p.set_defaults(fn=cmd_inspect)
    p = sub.add_parser("run", help="backtest a model (plus baselines) on a dataset")
    p.add_argument("--model", default="timestral-5m")
    p.add_argument("--dataset", default="multi_seasonality", help="synthetic scenario id, or file:<path>")
    p.add_argument("--targets", help="comma-separated target columns (file datasets)")
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0")
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--max-series", type=int)
    p.add_argument("--max-context", type=int)
    p.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    p.add_argument("--no-covariates", action="store_true")
    p.add_argument("--no-symmetric", action="store_true", help="symmetric_averaging=False (Timestral)")
    p.add_argument("--mask-flat-covariates", action="store_true", help="work around upstream issue #7")
    p.add_argument("--no-store", action="store_true")
    p.add_argument("--label")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("serve", help="start the local web app")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--tailscale", action="store_true", help="also listen on this PC's Tailscale address (tailnet only)")
    p.set_defaults(fn=cmd_serve)
    sub.add_parser("runs", help="list stored runs").set_defaults(fn=cmd_runs)
    p = sub.add_parser("show", help="print a stored run's metadata"); p.add_argument("run_id"); p.set_defaults(fn=cmd_show)
    args = parser.parse_args(argv)
    try:
        args.fn(args)
    except (ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
