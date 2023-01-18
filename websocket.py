import asyncio
import json
import websockets

async def handle_connection(websocket, path):
    # Receive data from the client
    data = await websocket.recv()
    print(json.loads(data))

start_server = websockets.serve(handle_connection, 'localhost', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()