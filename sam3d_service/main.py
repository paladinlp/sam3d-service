from __future__ import annotations

import uvicorn

from sam3d_service.config import Settings


if __name__ == "__main__":
    settings = Settings.from_env()
    uvicorn.run(
        "sam3d_service.app:app",
        host=settings.host,
        port=settings.port,
        workers=1,
    )
