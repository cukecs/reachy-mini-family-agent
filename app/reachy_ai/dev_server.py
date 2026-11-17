from __future__ import annotations

import hmac
import json
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .dev_tools import (
    CAMERA_PATH,
    ROOT,
    RUNTIME_STATE_PATH,
    DevInterfaceConfig,
    dev_diagnostics,
    dev_edit_file,
    dev_git_status,
    dev_list_files,
    dev_read_file,
    dev_run_tests,
    dev_search_code,
    dispatch_dev_command,
)
from .memory import MemoryStore

_WEB_UI_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Reachy AI 远程开发与调试控制台</title>
<style>
  body{margin:0;background:#0f141c;color:#e6edf5;font-family:system-ui,-apple-system,"Microsoft YaHei UI",sans-serif}
  header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between;padding:12px 18px;background:#161d27;border-bottom:1px solid #283447}
  header h1{margin:0;font-size:16px}
  .bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
  button{background:#233044;color:#e6edf5;border:1px solid #384a66;border-radius:6px;padding:6px 12px;cursor:pointer;font-size:13px}
  button.primary{background:#2f81f7;border-color:#2f81f7}
  button.danger{background:#a62b2b;border-color:#a62b2b}
  input,select,textarea{background:#0d131d;color:#e6edf5;border:1px solid #304057;border-radius:6px;padding:6px 8px;font-size:13px}
  main{display:grid;grid-template-columns:1.1fr 1fr;gap:12px;padding:12px;max-width:1440px;margin:auto}
  @media(max-width:960px){main{grid-template-columns:1fr}}
  .card{background:#161d27;border:1px solid #253042;border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:8px}
  .card h2{margin:0;font-size:14px;color:#9eb0c9}
  #cam{width:100%;max-height:320px;object-fit:contain;background:#090d14;border-radius:6px}
  pre,.box{background:#0d131d;border:1px solid #222d3f;border-radius:6px;padding:10px;margin:0;overflow:auto;font-family:Consolas,monospace;font-size:12px;line-height:1.45;max-height:300px;white-space:pre-wrap}
  .msg{margin-bottom:8px}.msg .meta{font-size:11px;color:#8fa1b9}.msg.user .txt{color:#79c0ff}.msg.assistant .txt{color:#7ee787}
  .row{display:flex;gap:8px;align-items:center}
  .row input{flex:1}
</style>
</head>
<body>
<header>
  <h1>Reachy AI 远程开发与调试控制台</h1>
  <div class="bar">
    <input id="token" type="password" placeholder="Bearer Token（本机可留空）" style="width:200px" />
    <button onclick="sendCmd('wake')">唤醒天线宝宝</button>
    <button onclick="sendCmd('sleep')">切回待机</button>
    <button class="danger" onclick="sendCmd('stop_speech')">立即停止说话</button>
    <button class="primary" onclick="runTests()">运行单元测试</button>
    <button onclick="loadDiag()">系统诊断</button>
  </div>
</header>
<main>
  <section class="card">
    <h2>实时画面与远程对话</h2>
    <img id="cam" alt="Camera offline" />
    <div id="transcript" class="box" style="height:220px"></div>
    <div class="row">
      <input id="prompt" placeholder="输入文本让机器人回答 (ask) 或直接朗读 (say)..." onkeydown="if(event.key==='Enter')sendPrompt('ask')" />
      <button class="primary" onclick="sendPrompt('ask')">远程提问</button>
      <button onclick="sendPrompt('say')">仅朗读</button>
    </div>
  </section>
  <section class="card">
    <h2>IDE 工作区浏览 / 搜索 / 诊断输出</h2>
    <div class="row">
      <input id="searchQ" placeholder="搜索代码（例如 speak / rename_user）" onkeydown="if(event.key==='Enter')searchCode()" />
      <button onclick="searchCode()">搜索代码</button>
      <button onclick="loadGit()">Git 状态</button>
    </div>
    <div class="row">
      <select id="subdir" onchange="listFiles()"><option value="app">app/</option><option value="scripts">scripts/</option><option value="tests">tests/</option><option value="logs">logs/</option><option value="docs">docs/</option></select>
      <select id="files" style="flex:1" onchange="readFile()"></select>
    </div>
    <pre id="output" style="height:330px">准备就绪。MCP 端点：POST /mcp</pre>
  </section>
</main>
<script>
const qs = new URLSearchParams(location.search);
if (qs.get('token')) document.getElementById('token').value = qs.get('token');
function hdrs(extra={}){
  const t = document.getElementById('token').value.trim();
  return t ? {...extra, 'Authorization':'Bearer '+t} : extra;
}
async function api(path, opts={}){
  const r = await fetch(path, {...opts, headers: hdrs(opts.headers||{})});
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
async function refreshLive(){
  const img = document.getElementById('cam');
  const t = document.getElementById('token').value.trim();
  img.src = '/dev/camera.jpg?ts=' + Date.now() + (t ? '&token=' + encodeURIComponent(t) : '');
  try{
    const st = await api('/dev/state');
    const box = document.getElementById('transcript');
    box.innerHTML = (st.messages||[]).slice(-18).map(m=>`<div class="msg ${m.role}"><div class="meta">${m.role} ${m.route||''} ${m.time||''}</div><div class="txt"></div></div>`).join('');
    const nodes = box.querySelectorAll('.txt');
    (st.messages||[]).slice(-18).forEach((m,i)=>{if(nodes[i])nodes[i].textContent=m.text;});
    box.scrollTop = box.scrollHeight;
  }catch(e){}
}
async function sendCmd(action, text=''){
  try{
    const res = await api('/dev/command', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action, text})});
    document.getElementById('output').textContent = JSON.stringify(res, null, 2);
  }catch(e){ document.getElementById('output').textContent = e.message; }
}
async function sendPrompt(mode){
  const inp = document.getElementById('prompt');
  const text = inp.value.trim();
  if(!text) return;
  inp.value = '';
  await sendCmd(mode, text);
  refreshLive();
}
async function listFiles(){
  try{
    const sub = document.getElementById('subdir').value;
    const data = await api('/dev/files?subdir=' + encodeURIComponent(sub));
    const sel = document.getElementById('files');
    sel.innerHTML = data.files.map(f=>`<option value="${f.path}">${f.path} (${f.bytes}B)</option>`).join('');
    if(data.files.length) readFile();
  }catch(e){ document.getElementById('output').textContent = e.message; }
}
async function readFile(){
  const path = document.getElementById('files').value;
  if(!path) return;
  try{
    const data = await api('/dev/file?path=' + encodeURIComponent(path) + '&start_line=1&end_line=220');
    document.getElementById('output').textContent = data.content;
  }catch(e){ document.getElementById('output').textContent = e.message; }
}
async function searchCode(){
  const q = document.getElementById('searchQ').value.trim();
  if(!q) return;
  try{
    const data = await api('/dev/search?q=' + encodeURIComponent(q) + '&subdir=.');
    document.getElementById('output').textContent = (data.matches||[]).map(m=>`${m.path}:${m.line}: ${m.text}`).join('\\n') || '无匹配';
  }catch(e){ document.getElementById('output').textContent = e.message; }
}
async function runTests(){
  document.getElementById('output').textContent = '正在运行单元测试...';
  try{
    const data = await api('/dev/tests', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    document.getElementById('output').textContent = data.output;
  }catch(e){ document.getElementById('output').textContent = e.message; }
}
async function loadDiag(){
  try{
    const data = await api('/dev/diagnostics');
    document.getElementById('output').textContent = JSON.stringify(data, null, 2);
  }catch(e){ document.getElementById('output').textContent = e.message; }
}
async function loadGit(){
  try{
    const data = await api('/dev/git');
    document.getElementById('output').textContent = JSON.stringify(data, null, 2);
  }catch(e){ document.getElementById('output').textContent = e.message; }
}
listFiles();
refreshLive();
setInterval(refreshLive, 1500);
</script>
</body>
</html>
"""


def mcp_tools_catalog(config: DevInterfaceConfig) -> list[dict[str, Any]]:
    tools = [
        {
            "name": "reachy_diagnostics",
            "description": "Inspect live Reachy daemon, router, CPU/RAM, processes, camera freshness, and runtime state.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "reachy_list_files",
            "description": "List allowed source, test, script, doc, or log files inside the Reachy workspace.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subdir": {"type": "string", "default": "app"},
                    "pattern": {"type": "string", "default": "*"},
                },
            },
        },
        {
            "name": "reachy_read_file",
            "description": "Read a line range from an allowed Reachy source or log file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "default": 1},
                    "end_line": {"type": "integer", "default": 160},
                },
                "required": ["path"],
            },
        },
        {
            "name": "reachy_search_code",
            "description": "Search code in the Reachy workspace by regex or keyword.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "subdir": {"type": "string", "default": "app"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "reachy_git_status",
            "description": "Inspect git branch status and recent commits in D:\\ReachyAI.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "reachy_run_tests",
            "description": "Run the unittest suite in D:\\ReachyAI\\tests.",
            "inputSchema": {
                "type": "object",
                "properties": {"pattern": {"type": "string", "default": ""}},
            },
        },
        {
            "name": "reachy_profiles_list",
            "description": "List enrolled household profiles with face/voice template counts (no raw biometrics).",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    ]
    if config.allow_code_edit:
        tools.append(
            {
                "name": "reachy_edit_file",
                "description": "Replace an exact text block inside an allowed source/script/test/doc file.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "target": {"type": "string"},
                        "replacement": {"type": "string"},
                    },
                    "required": ["path", "target", "replacement"],
                },
            }
        )
    if config.allow_remote_control:
        tools.append(
            {
                "name": "reachy_send_command",
                "description": "Send a remote command to the live Reachy agent: say, ask, stop_speech, wake, or sleep.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["say", "ask", "stop_speech", "wake", "sleep"]},
                        "text": {"type": "string", "default": ""},
                    },
                    "required": ["action"],
                },
            }
        )
    return tools


def execute_mcp_tool(
    name: str,
    arguments: dict[str, Any],
    config: DevInterfaceConfig,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    if name == "reachy_diagnostics":
        return dev_diagnostics(root=root)
    if name == "reachy_list_files":
        return dev_list_files(
            str(arguments.get("subdir", "app")),
            str(arguments.get("pattern", "*")),
            root=root,
        )
    if name == "reachy_read_file":
        return dev_read_file(
            str(arguments["path"]),
            int(arguments.get("start_line", 1)),
            int(arguments.get("end_line", 160)),
            root=root,
        )
    if name == "reachy_search_code":
        return dev_search_code(
            str(arguments["query"]),
            str(arguments.get("subdir", "app")),
            root=root,
        )
    if name == "reachy_git_status":
        return dev_git_status(root=root)
    if name == "reachy_run_tests":
        return dev_run_tests(str(arguments.get("pattern", "")), root=root)
    if name == "reachy_profiles_list":
        db_path = root / "data" / "memory.sqlite3"
        store = MemoryStore(db_path, official_memory_root=None)
        try:
            return {"ok": True, "profiles": store.list_profiles()}
        finally:
            store.close()
    if name == "reachy_edit_file":
        return dev_edit_file(
            str(arguments["path"]),
            str(arguments["target"]),
            str(arguments["replacement"]),
            allow_code_edit=config.allow_code_edit,
            root=root,
        )
    if name == "reachy_send_command":
        if not config.allow_remote_control:
            raise PermissionError("远程控制已被 dev_interface.allow_remote_control 禁用")
        return dispatch_dev_command(
            str(arguments["action"]),
            str(arguments.get("text", "")),
            root=root,
        )
    raise ValueError(f"未知 MCP 工具：{name}")


class DevDebugHandler(BaseHTTPRequestHandler):
    dev_config: DevInterfaceConfig
    dev_token: str
    workspace_root: Path = ROOT

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[DEV] {self.client_address[0]} - {fmt % args}")

    def _is_authorized(self, query_params: dict[str, list[str]]) -> bool:
        client_ip = str(self.client_address[0])
        if client_ip in {"127.0.0.1", "::1"}:
            return True
        auth = str(self.headers.get("Authorization", "")).strip()
        provided = ""
        if auth.lower().startswith("bearer "):
            provided = auth[7:].strip()
        elif self.headers.get("X-Reachy-Token"):
            provided = str(self.headers.get("X-Reachy-Token", "")).strip()
        elif "token" in query_params and query_params["token"]:
            provided = str(query_params["token"][0]).strip()
        return bool(provided and hmac.compare_digest(provided, self.dev_token))

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path in {"/", "/index.html"}:
            self._send_html(HTTPStatus.OK, _WEB_UI_HTML)
            return
        if not self._is_authorized(query):
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized; Bearer token required"})
            return
        try:
            if parsed.path == "/dev/diagnostics":
                self._send_json(HTTPStatus.OK, dev_diagnostics(root=self.workspace_root))
            elif parsed.path == "/dev/state":
                state_path = self.workspace_root / "data" / "runtime-state.json"
                data = (
                    json.loads(state_path.read_text(encoding="utf-8"))
                    if state_path.exists()
                    else {"status": "offline", "messages": []}
                )
                self._send_json(HTTPStatus.OK, data)
            elif parsed.path == "/dev/camera.jpg":
                cam_path = self.workspace_root / "data" / "live-camera.jpg"
                if not cam_path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND, "camera frame unavailable")
                    return
                raw = cam_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif parsed.path == "/dev/files":
                subdir = query.get("subdir", ["."])[0]
                pattern = query.get("pattern", ["*"])[0]
                self._send_json(
                    HTTPStatus.OK,
                    dev_list_files(subdir, pattern, root=self.workspace_root),
                )
            elif parsed.path == "/dev/file":
                path = query.get("path", [""])[0]
                start = int(query.get("start_line", ["1"])[0])
                end = int(query.get("end_line", ["160"])[0])
                self._send_json(
                    HTTPStatus.OK,
                    dev_read_file(path, start, end, root=self.workspace_root),
                )
            elif parsed.path == "/dev/search":
                q = query.get("q", [""])[0]
                subdir = query.get("subdir", ["app"])[0]
                self._send_json(
                    HTTPStatus.OK,
                    dev_search_code(q, subdir, root=self.workspace_root),
                )
            elif parsed.path == "/dev/git":
                self._send_json(HTTPStatus.OK, dev_git_status(root=self.workspace_root))
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except PermissionError as exc:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
        except (ValueError, FileNotFoundError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if not self._is_authorized(query):
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized; Bearer token required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if parsed.path == "/dev/tests":
                self._send_json(
                    HTTPStatus.OK,
                    dev_run_tests(str(payload.get("pattern", "")), root=self.workspace_root),
                )
            elif parsed.path == "/dev/edit":
                self._send_json(
                    HTTPStatus.OK,
                    dev_edit_file(
                        str(payload.get("path", "")),
                        str(payload.get("target", "")),
                        str(payload.get("replacement", "")),
                        allow_code_edit=self.dev_config.allow_code_edit,
                        root=self.workspace_root,
                    ),
                )
            elif parsed.path == "/dev/command":
                if not self.dev_config.allow_remote_control:
                    raise PermissionError("远程控制已被禁用")
                self._send_json(
                    HTTPStatus.OK,
                    dispatch_dev_command(
                        str(payload.get("action", "")),
                        str(payload.get("text", "")),
                        root=self.workspace_root,
                    ),
                )
            elif parsed.path == "/mcp":
                self._handle_mcp(payload)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except PermissionError as exc:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": str(exc)})
        except (ValueError, json.JSONDecodeError, FileNotFoundError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def _handle_mcp(self, rpc: dict[str, Any]) -> None:
        rpc_id = rpc.get("id")
        method = str(rpc.get("method", ""))
        params = rpc.get("params") or {}
        if method == "initialize":
            self._send_json(
                HTTPStatus.OK,
                {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "reachy-ai-dev", "version": "1.0.0"},
                    },
                },
            )
        elif method == "notifications/initialized":
            self.send_response(HTTPStatus.ACCEPTED)
            self.end_headers()
        elif method == "ping":
            self._send_json(HTTPStatus.OK, {"jsonrpc": "2.0", "id": rpc_id, "result": {}})
        elif method == "tools/list":
            self._send_json(
                HTTPStatus.OK,
                {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {"tools": mcp_tools_catalog(self.dev_config)},
                },
            )
        elif method == "tools/call":
            tool_name = str(params.get("name", ""))
            tool_args = dict(params.get("arguments") or {})
            try:
                result = execute_mcp_tool(
                    tool_name,
                    tool_args,
                    self.dev_config,
                    root=self.workspace_root,
                )
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, ensure_ascii=False, indent=2),
                                }
                            ],
                            "isError": False,
                        },
                    },
                )
            except Exception as exc:
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "result": {
                            "content": [{"type": "text", "text": str(exc)}],
                            "isError": True,
                        },
                    },
                )
        else:
            self._send_json(
                HTTPStatus.OK,
                {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                },
            )


def start_dev_server_background(
    config: DevInterfaceConfig, *, root: Path = ROOT
) -> ThreadingHTTPServer | None:
    if not config.enabled:
        return None
    token = config.resolve_token()
    handler_cls = type(
        "_BoundDevHandler",
        (DevDebugHandler,),
        {
            "dev_config": config,
            "dev_token": token,
            "workspace_root": root,
        },
    )
    server = ThreadingHTTPServer((config.listen_host, config.listen_port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(
        f"Reachy Dev & MCP interface: http://{config.listen_host}:{config.listen_port} "
        f"(MCP: /mcp, code_edit={config.allow_code_edit})"
    )
    return server
