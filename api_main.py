"""Start the private time-series experiment API."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("experiment_api.app:app", host="127.0.0.1", port=8000, reload=False)

