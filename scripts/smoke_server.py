"""Start a temporary headless Streamlit server, verify HTTP health, then stop it."""
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryFile(mode='w+b') as log:
        proc = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app/app.py'),
            '--server.headless', 'true', '--server.address', '127.0.0.1',
            '--server.port', str(port), '--browser.gatherUsageStats', 'false',
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(60):
                if proc.poll() is not None:
                    break
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=2) as r:
                        if r.status == 200 and r.read().strip() == b'ok':
                            with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=2) as home:
                                assert home.status == 200
                            print('Streamlit startup: homepage and health endpoint HTTP 200')
                            return
                except OSError:
                    pass
                time.sleep(0.5)
            log.seek(0)
            raise RuntimeError(log.read().decode('utf-8', errors='replace'))
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == '__main__':
    main()
