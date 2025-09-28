import { AppServer, AppSession } from "@mentra/sdk";
import express from "express";
import http from "http";
import { WebSocketServer } from "ws";


const app = express();
app.use(express.json());

const server = http.createServer(app);

const sessions = new Map<string, AppSession>();

app.post("/receive", (req, res) => {
  const text = req.body.name; // expecting { "text": "..." }
  console.log("Received HTTP text:", text);

  for (const [id, session] of sessions.entries()) {
    session.logger.info(`📩 Received HTTP text for session ${id}: ${text}`);
    session.layouts.showTextWall(text);
  }

  res.json({ status: "ok", received: text });
});


const wss = new WebSocketServer({ server });

wss.on("connection", (ws) => {
  console.log("🌐 WebSocket client connected");

  ws.on("message", (msg) => {
    try {
      const data = JSON.parse(msg.toString());
      const { ts, risk, direction } = data;

      console.log("Received WS data:", data);
      
      const text = `⚠️ Risk: ${risk}\n➡️ Direction: ${direction}`;

      for (const [id, session] of sessions.entries()) {

        session.logger.info(text)
        session.logger.info(`📩 WS update for session ${id}: ${text}`);
        session.layouts.showTextWall(text);
      }
    } catch (err) {
      console.error("Invalid WS message:", err);
    }
  });

  ws.on("close", () => {
    console.log("❌ WebSocket client disconnected");
  });
});

server.listen(3000, () => {
  console.log("HTTP + WS server listening on :3000");
});

const PACKAGE_NAME = process.env.PACKAGE_NAME || "com.example.myfirstmentraosapp";
const PORT = parseInt(process.env.PORT || "8080");
const MENTRAOS_API_KEY = process.env.MENTRAOS_API_KEY!;

class MyMentraOSApp extends AppServer {
  protected async onSession(session: AppSession, sessionId: string, userId: string) {
    session.logger.info(`New session ${sessionId} for user ${userId}`);
    console.log("New session", sessionId, userId);

    sessions.set(sessionId, session);

    await session.layouts.showTextWall("👋 Hello from MentraOS!");

    session.events.onDisconnected(() => {
      sessions.delete(sessionId);
      session.logger.info(`Session ${sessionId} disconnected.`);
    });
  }
}

new MyMentraOSApp({
  packageName: PACKAGE_NAME,
  apiKey: MENTRAOS_API_KEY,
  port: PORT,
})
  .start()
  .catch((err) => {
    console.error("Failed to start Mentra app:", err);
  });
