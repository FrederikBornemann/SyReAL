import asyncio
import json
import websockets

async def send_data():
    # Connect to the websocket
    async with websockets.connect('ws://localhost:8765') as websocket:
        # Send some data
        data = {'key': 'value'}
        await websocket.send(json.dumps(data))

# Start the async task
asyncio.get_event_loop().run_until_complete(send_data())