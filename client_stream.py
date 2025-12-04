# client_stream.py

import cv2
import base64
import asyncio
import websockets
import numpy as np

# SERVER_URL = "ws://172.16.203.101:8000/ws"
SERVER_URL = "ws://localhost:8000/ws"


async def stream():
    vid = cv2.VideoCapture(0)

    async with websockets.connect(SERVER_URL) as ws:
        while True:
            ret, frame = vid.read()
            if not ret:
                continue

            _, buffer = cv2.imencode(".jpg", frame)

            # await ws.send(base64.b64encode(buffer.tobytes()).decode())

            # # processed = await ws.receive()
            # processed = await ws.recv()

            # img_data = base64.b64decode(processed)

            await ws.send(buffer.tobytes())
            processed = await ws.recv()
            img_data = processed


            arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            cv2.imshow("DGX Output", img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    vid.release()
    cv2.destroyAllWindows()

asyncio.run(stream())
