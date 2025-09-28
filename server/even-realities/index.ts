import { AppServer, AppSession } from "@mentra/sdk";
import express from "express";
import fs from "fs";

// -------------------
// Express HTTP server
// -------------------
const app = express();
app.use(express.json());

// Keep a map of active sessions so we can update them from HTTP
const sessions = new Map<string, AppSession>();

app.post("/receive", (req, res) => {
  const text  = req.body.name; // expecting { "text": "..." }

  // Push to all active sessions (or pick one)
  for (const [id, session] of sessions.entries()) {
    session.logger.info(`📩 Received HTTP text for session ${id}: ${text}`);
    session.layouts.showTextWall(text);
  }

  res.json({ status: "ok", received: text });
});

app.listen(3000, () => {
  console.log("HTTP server listening on :3000");
});

// -------------------
// MentraOS App Server
// -------------------
const PACKAGE_NAME = process.env.PACKAGE_NAME || "com.example.myfirstmentraosapp";
const PORT = parseInt(process.env.PORT || "8080");
const MENTRAOS_API_KEY = process.env.MENTRAOS_API_KEY!;

class MyMentraOSApp extends AppServer {
  protected async onSession(session: AppSession, sessionId: string, userId: string) {
    session.logger.info(`New session ${sessionId} for user ${userId}`);

    // Track the session
    sessions.set(sessionId, session);

    // Show a default screen
    await session.layouts.showTextWall("👋 Hello from MentraOS!");

    // Cleanup on disconnect
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
}).start().catch((err) => {
  console.error("Failed to start Mentra app:", err);
});