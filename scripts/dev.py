"""Run the API and web client together, and stop both when either exits."""

import os
import signal
import subprocess
import sys
import time


def main() -> int:
    uv = os.environ.get("UV", "uv")
    npm = os.environ.get("NPM", "npm")
    api_host = os.environ.get("API_HOST", "127.0.0.1")
    api_port = os.environ.get("API_PORT", "8000")
    web_host = os.environ.get("WEB_HOST", "127.0.0.1")
    web_port = os.environ.get("WEB_PORT", "5173")
    api_base_url = os.environ.get("API_BASE_URL", f"http://{api_host}:{api_port}/api/v1")

    commands = [
        (
            "backend",
            [
                uv,
                "run",
                "--directory",
                "backend",
                "fastapi",
                "dev",
                "--host",
                api_host,
                "--port",
                api_port,
            ],
        ),
        (
            "frontend",
            [
                npm,
                "--prefix",
                "frontend",
                "run",
                "dev",
                "--",
                "--host",
                web_host,
                "--port",
                web_port,
            ],
        ),
    ]
    processes: list[tuple[str, subprocess.Popen[bytes]]] = []

    def stop_processes() -> None:
        for _, process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _, process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

    def handle_signal(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, handle_signal)
    try:
        for name, command in commands:
            print(f"Starting {name}: {' '.join(command)}", flush=True)
            process_env = os.environ.copy()
            if name == "frontend":
                process_env["VITE_API_BASE_URL"] = api_base_url
            processes.append(
                (
                    name,
                    subprocess.Popen(command, env=process_env, start_new_session=True),
                )
            )

        while True:
            for name, process in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    print(f"{name} stopped with exit code {exit_code}; stopping the stack.")
                    return exit_code
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopping EnerCol development stack…")
        return 0
    finally:
        stop_processes()


if __name__ == "__main__":
    sys.exit(main())
