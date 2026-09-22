from __future__ import annotations

import argparse
import os

from experiment_api.app import _default_service
from experiment_mcp.server import StdioMcpServer
from experiment_mcp.tools import ExperimentTools


def main() -> None:
    parser = argparse.ArgumentParser(description="CrewAI forecasting MCP stdio server")
    parser.add_argument("--data-root", help="Experiment storage directory")
    args = parser.parse_args()
    if args.data_root:
        os.environ["EXPERIMENT_API_DATA"] = args.data_root
    StdioMcpServer(ExperimentTools(_default_service())).run()


if __name__ == "__main__":
    main()
