# viewer.py
import argparse, sys, time, asyncio
import cv2, numpy as np
import requests
import websockets  # pip install websockets

def try_videocap(url, title):
    print(f"[viewer] trying VideoCapture on {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("[viewer] VideoCapture could not open stream.", file=sys.stderr)
        return False
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    last, shown = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02); continue
            cv2.imshow(title, frame); shown += 1
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
            if time.time() - last > 2:
                print(f"[viewer] frames shown (VideoCapture): {shown}")
                shown, last = 0, time.time()
    finally:
        cap.release(); 
        try: cv2.destroyWindow(title)
        except: pass
    return True

async def run_ws(url, title):
    print(f"[viewer] connecting WebSocket {url}")
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    shown, last = 0, time.time()
    async with websockets.connect(url, max_size=None) as ws:
        while True:
            jpg = await ws.recv()  # bytes
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            cv2.imshow(title, img); shown += 1
            if (cv2.waitKey(1) & 0xFF) == ord('q'): break
            if time.time() - last > 2:
                print(f"[viewer] frames shown (WebSocket): {shown}")
                shown, last = 0, time.time()
    try: cv2.destroyWindow(title)
    except: pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Use ws:// or wss:// for WebSocket, or http(s):// for MJPEG")
    ap.add_argument("--title", default="WebRTC Camera Stream")
    args = ap.parse_args()

    if args.url.startswith("ws://") or args.url.startswith("wss://"):
        asyncio.run(run_ws(args.url, args.title))
    else:
        ok = try_videocap(args.url, args.title)
        if not ok:
            print("[viewer] fallback to parser disabled on purpose; prefer WS for low latency.", file=sys.stderr)

if __name__ == "__main__":
    main()
