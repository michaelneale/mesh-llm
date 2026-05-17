#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from urllib import request


def fetch_json(url: str):
    with request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode())


def fetch_text(url: str, data: bytes | None = None):
    req = request.Request(url, data=data, headers={"content-type": "application/json"})
    with request.urlopen(req, timeout=90) as response:
        return response.read().decode()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local mesh-llm config smoke against a rewritten fixture.")
    parser.add_argument('--binary', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--draft-path')
    parser.add_argument('--mmproj-path')
    parser.add_argument('--api-port', type=int, default=9437)
    parser.add_argument('--console-port', type=int, default=3231)
    parser.add_argument('--max-wait', type=int, default=180)
    args = parser.parse_args()

    fixture = Path(args.fixture)
    text = fixture.read_text()
    text = text.replace('__LOCAL_MODEL_PATH__', args.model_path)
    text = text.replace('__LOCAL_INCOMPATIBLE_DRAFT_PATH__', args.draft_path or args.model_path)
    text = text.replace('__LOCAL_MMPROJ_PATH__', args.mmproj_path or '__LOCAL_MMPROJ_PATH__')

    tmp = tempfile.NamedTemporaryFile('w', suffix='.toml', delete=False)
    tmp.write(text)
    tmp.flush()
    tmp.close()
    tmp_path = Path(tmp.name)

    cmd = [
        args.binary,
        'serve',
        '--config',
        str(tmp_path),
        '--headless',
        '--port',
        str(args.api_port),
        '--console',
        str(args.console_port),
    ]
    log_file = tempfile.NamedTemporaryFile('w+', suffix='.log', delete=False)
    log_path = Path(log_file.name)
    log_file.close()
    proc = subprocess.Popen(cmd, stdout=log_path.open('w'), stderr=subprocess.STDOUT, text=True)
    try:
        ready = False
        status = None
        for _ in range(args.max_wait):
            if proc.poll() is not None:
                break
            try:
                status = fetch_json(f'http://127.0.0.1:{args.console_port}/api/status')
                if status.get('llama_ready'):
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1)
        if not ready:
            raise RuntimeError('runtime did not become ready')

        models = None
        model_id = None
        for _ in range(60):
            models = fetch_json(f'http://127.0.0.1:{args.api_port}/v1/models')
            data = models.get('data') or []
            if data:
                model_id = data[0].get('id')
                break
            time.sleep(1)
        if not model_id:
            raise RuntimeError('v1/models did not publish a model')

        payload = json.dumps({
            'model': model_id,
            'messages': [{'role': 'user', 'content': 'Say hello in exactly three words.'}],
            'max_tokens': 4,
            'temperature': 0,
        }).encode()
        chat = fetch_text(f'http://127.0.0.1:{args.api_port}/v1/chat/completions', payload)

        print('COMMAND:', ' '.join(cmd))
        print('FIXTURE:', fixture)
        print('TEMP_CONFIG:', tmp_path)
        print('STATUS_JSON:', json.dumps(status, indent=2, sort_keys=True))
        print('MODELS_JSON:', json.dumps(models, indent=2, sort_keys=True))
        print('CHAT_JSON:', chat)
        print('LOG_PATH:', log_path)
        print('LOG_LINES_BEGIN')
        print(''.join(log_path.read_text().splitlines(True)[-80:]))
        print('LOG_LINES_END')
        return 0
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    raise SystemExit(main())
