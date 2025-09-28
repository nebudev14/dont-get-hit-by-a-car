import { AppServer, AppSession } from "@mentra/sdk";
import express from "express";

const fs = require('fs');
const app = express();
let container = "";
app.use(express.json());

app.get("/receive", (req, res) => {
  console.log("req received: " + req.body.name);
  // container = 
  res.json({ status: "success", received: req.body});
})

app.listen(3000, () => {
  console.log("server running on 3000")
})

/**
 * A custom keyword that triggers our action once detected in speech
 */
const ACTIVATION_PHRASE = "computer";
const ACTIVATION_PHRASE1 = "accident";
const ACTIVATION_PHRASE2 = "clear view";

/**
 * VoiceActivationServer – an App that listens for final transcriptions and
 * reacts when the user utters the ACTIVATION_PHRASE.
 */
class VoiceActivationServer extends AppServer {
  /**
   * onSession is called automatically whenever a user connects.
   *
   * @param session   – Connection-scoped helper APIs and event emitters
   * @param sessionId – Unique identifier for this connection
   * @param userId    – MentraOS user identifier
   */
  protected async onSession(
    session: AppSession,
    sessionId: string,
    userId: string,
  ): Promise<void> {
    session.logger.info(`🔊  Session ${sessionId} started for ${userId}`);

    // 1️⃣  Subscribe to speech transcriptions
    const unsubscribe = session.events.onTranscription((data) => {
      // 2️⃣  Ignore interim results – we only care about the final text
      if (!data.isFinal) return;

      // 3️⃣  Normalize casing & whitespace for a simple comparison
      const spokenText = data.text.toLowerCase().trim();
      session.logger.debug(`Heard: "${spokenText}"`);

      // 4️⃣  Check for the activation phrase
      if (spokenText.includes(ACTIVATION_PHRASE)) {
        session.logger.info("✨ Activation phrase detected!");

        // 5️⃣  Do something useful – here we show a text overlay
        session.layouts.showTextWall("👋 How can I help? Hello hello");
      }
      if (spokenText.includes(ACTIVATION_PHRASE1)) {
        session.logger.info("✨ Activation phrase 1 detected!");

        // image
        const imageBuffer = fs.readFileSync('./warningIconBMP2.bmp');
        const base64data = imageBuffer.toString('base64');
        session.layouts.showBitmapView(base64data);
      }
      if (spokenText.includes(ACTIVATION_PHRASE2)) {
        session.logger.info("✨ Activation phrase 2 detected!");
        session.layouts.clearView();
      }
    });

    // 6️⃣  Clean up the listener when the session ends
    this.addCleanupHandler(unsubscribe);
  }
}

// Bootstrap the server using environment variables for configuration
new VoiceActivationServer({
  packageName: process.env.PACKAGE_NAME ?? "com.example.voiceactivation",
  apiKey: process.env.MENTRAOS_API_KEY!,
  port: Number(process.env.PORT ?? "3000"),
}).start();