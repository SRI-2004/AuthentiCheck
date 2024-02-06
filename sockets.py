import asyncio
import websockets

async def handle_client(websocket, path):
    print(f"Client connected: {websocket.remote_address}")

    try:
        while True:
            # Receive data from the client
            data = await websocket.recv()
            print(f"Received data: {data}")

            # Process the received data and generate a response
            response_data = f"Server processed: {data}"

            # Send the response back to the client
            await websocket.send(response_data)
    except websockets.exceptions.ConnectionClosedError:
        print(f"Client disconnected: {websocket.remote_address}")

async def main():
    # Define the host and port for your WebSocket server
    host = '0.0.0.0'  # localhost
    port = 8080

    # Start the WebSocket server
    server = await websockets.serve(handle_client, host, port)
    print(f"WebSocket server listening on {host}:{port}")

    # Keep the server running
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
