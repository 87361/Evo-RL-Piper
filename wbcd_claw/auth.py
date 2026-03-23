"""Simple password-based auth middleware for WBCDClaw."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware

from wbcd_claw.config import AppConfig

_LOGIN_PAGE = """<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'/>
<meta name='viewport' content='width=device-width,initial-scale=1'/>
<title>WBCDClaw - Login</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#111;color:#eee;font-family:system-ui,-apple-system,sans-serif;
  display:flex;align-items:center;justify-content:center;height:100vh}
.box{background:#1a1a2e;padding:32px;border-radius:12px;width:min(340px,90vw)}
h2{text-align:center;margin-bottom:20px;font-size:20px}
input{width:100%;padding:14px;margin-bottom:16px;border:1px solid #333;
  border-radius:8px;background:#222;color:#eee;font-size:16px}
button{width:100%;padding:14px;border:none;border-radius:8px;
  background:#4361ee;color:#fff;font-size:16px;cursor:pointer}
button:active{background:#3a56d4}
.err{color:#ff6b6b;text-align:center;margin-bottom:12px;font-size:14px}
</style></head><body>
<div class='box'>
<h2>WBCDClaw</h2>
<div id='err' class='err'></div>
<form onsubmit='return doLogin()'>
<input id='pw' type='password' placeholder='Password' autofocus/>
<button type='submit'>Login</button>
</form>
</div>
<script>
async function doLogin(){
  const pw=document.getElementById('pw').value;
  const r=await fetch('/api/auth/login',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({password:pw})});
  const d=await r.json();
  if(d.ok){location.href='/';}
  else{document.getElementById('err').innerText=d.error||'Wrong password';}
  return false;
}
</script></body></html>"""


def _make_token(password: str, secret: str) -> str:
    return hmac.new(secret.encode(), password.encode(), hashlib.sha256).hexdigest()


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, config: AppConfig, secret: str | None = None):
        super().__init__(app)
        self.config = config
        self.secret = secret or secrets.token_hex(32)
        self.valid_token = _make_token(config.password, self.secret)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.config.password:
            return await call_next(request)

        path = request.url.path
        if path == "/api/auth/login" or path == "/login":
            return await call_next(request)

        token = request.cookies.get(self.config.cookie_name, "")
        if hmac.compare_digest(token, self.valid_token):
            return await call_next(request)

        if path.startswith("/api/"):
            return Response(
                content='{"ok":false,"error":"unauthorized"}',
                status_code=401,
                media_type="application/json",
            )
        return HTMLResponse(_LOGIN_PAGE)


def register_auth_routes(app, config: AppConfig, secret: str):
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    valid_token = _make_token(config.password, secret)

    class LoginPayload(BaseModel):
        password: str

    @app.post("/api/auth/login")
    def login(payload: LoginPayload) -> Response:
        if not config.password:
            return JSONResponse({"ok": True})
        if not hmac.compare_digest(payload.password, config.password):
            return JSONResponse({"ok": False, "error": "wrong password"}, status_code=401)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(
            config.cookie_name,
            valid_token,
            max_age=config.cookie_max_age,
            httponly=True,
            samesite="lax",
        )
        return resp
