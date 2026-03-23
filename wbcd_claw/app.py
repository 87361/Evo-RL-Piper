"""FastAPI application factory for WBCDClaw."""

from __future__ import annotations

import secrets

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from wbcd_claw.auth import AuthMiddleware, register_auth_routes
from wbcd_claw.config import AppConfig
from wbcd_claw.pages import PAGE
from wbcd_claw.sample_api import init_sample_state, router as sample_router
from wbcd_claw.train_api import init_train_state, router as train_router
from wbcd_claw.train_manager import TrainManager


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="WBCDClaw")

    auth_secret = secrets.token_hex(32)
    if config.password:
        app.add_middleware(AuthMiddleware, config=config, secret=auth_secret)
        register_auth_routes(app, config, auth_secret)

    init_sample_state(config)
    app.include_router(sample_router)

    manager = TrainManager(config)
    init_train_state(manager)
    app.include_router(train_router)

    for entry in config.datasets:
        if entry.video_root.exists():
            app.mount(
                f"/media/{entry.name}",
                StaticFiles(directory=str(entry.video_root)),
                name=f"media_{entry.name}",
            )

    @app.get("/", response_class=HTMLResponse)
    def home():
        return PAGE

    return app
