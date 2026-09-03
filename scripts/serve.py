"""启动 Research Assistant API 服务。

用法:
    python scripts/serve.py            # http://0.0.0.0:8000
    python scripts/serve.py --port 9000
"""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description='启动 Research Assistant API')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run('research_assistant.api:app', host=args.host, port=args.port)


if __name__ == '__main__':
    main()
