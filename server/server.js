// server/server.js
require("dotenv").config();
const express = require("express");
const path = require("path");
const http = require("http");
const { spawn } = require("child_process");
const WebSocket = require("ws");

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PORT    = process.env.PORT || 3000;
const PYTHON  = process.env.PYTHON || "python";                   // Windows면 "py"도 가능
const AI_ROOT = path.resolve(process.env.AI_ROOT || path.join(__dirname, "..", "ai")); // main.py가 있는 폴더

// ──────────────────────────────────────────────
// 정적 파일: front/html, front/css, front/js 매핑
// http://localhost:3000/chat.html 로 접속
// ──────────────────────────────────────────────
app.use("/",     express.static(path.resolve(__dirname, "..", "front", "html")));
app.use("/css",  express.static(path.resolve(__dirname, "..", "front", "css")));
app.use("/js",   express.static(path.resolve(__dirname, "..", "front", "js")));

app.get("/ping", (_, res) => res.json({ ok: true }));

wss.on("connection", (ws) => {
  console.log("✅ WebSocket connected");

  // -u: 파이썬 stdout 무버퍼(즉시 송출)
  const py = spawn(PYTHON, ["-u", "main.py"], {
    cwd: AI_ROOT,
    env: { ...process.env, PYTHONIOENCODING: "utf-8" },
    stdio: ["pipe", "pipe", "pipe"],
  });

  let outBuf = "", errBuf = "";

  const flushLines = (buf, sendFn) => {
    const parts = buf.split(/\r?\n/);
    for (let i = 0; i < parts.length - 1; i++) {
      const line = parts[i];
      if (line.trim() !== "") sendFn(line);
    }
    return parts[parts.length - 1]; // 마지막 덜 온 조각은 유지
  };

  py.stdout.on("data", (chunk) => {
    outBuf += chunk.toString("utf8");
    outBuf = flushLines(outBuf, (line) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(line);
    });
  });

  py.stderr.on("data", (chunk) => {
    errBuf += chunk.toString("utf8");
    errBuf = flushLines(errBuf, (line) => {
      if (ws.readyState === WebSocket.OPEN) ws.send("⚠️ " + line);
    });
  });

  py.on("close", (code) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.close();
    }
  });

  ws.on("message", (msg) => {
    try {
      // 브라우저 입력 → 파이썬 input()
      py.stdin.write(msg.toString() + "\n");
    } catch (_) {}
  });

  ws.on("close", () => { try { py.kill(); } catch (_) {} });
});

server.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
  console.log(`   Python cwd: ${AI_ROOT}`);
});
