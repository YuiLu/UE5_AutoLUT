#!/usr/bin/env python3
"""
AutoLUT WebSocket Server
This script receives video data from the UE5 AutoLUT plugin and processes it.

Requirements:
    pip install websockets

Usage:
    python autolut_server.py

The server listens on ws://127.0.0.1:8765 by default.
"""

import asyncio
import websockets
import json
import base64
import os
import traceback
from datetime import datetime

# Configuration
HOST = "127.0.0.1"
PORT = 8765
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "received_videos")

# Connection counter for debugging
connection_counter = 0


def log(msg: str):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}")


async def process_video(video_data: bytes, filename: str, total_frames: int, fps: int) -> dict:
    """
    Process the received video.
    
    Args:
        video_data: Raw video bytes (MP4)
        filename: Original filename
        total_frames: Number of frames in the video
        fps: Frames per second
    """
    log(f"Processing video: {filename}")
    log(f"Video size: {len(video_data)} bytes")
    log(f"Total frames: {total_frames}, FPS: {fps}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"video_{timestamp}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Save the video
    log(f"Saving video to: {output_path}")
    with open(output_path, 'wb') as f:
        f.write(video_data)
    
    log(f"Video saved successfully!")
    
    # Calculate video duration
    duration = total_frames / fps if fps > 0 else 0
    
    result = {
        "status": "success",
        "message": f"Video received and saved ({len(video_data) / 1024 / 1024:.2f} MB, {duration:.1f}s)",
        "video_path": output_path,
        "video_size_mb": round(len(video_data) / 1024 / 1024, 2),
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2)
    }
    
    # TODO: Add your video processing logic here
    # For example: analyze video, generate LUT, etc.
    
    return result


async def process_images(screenshot_data: bytes, source_image_data: bytes = None) -> dict:
    """
    Process the received images (legacy support).
    """
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "received_images")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    screenshot_path = os.path.join(output_dir, f"screenshot_{timestamp}.png")
    with open(screenshot_path, 'wb') as f:
        f.write(screenshot_data)
    log(f"Screenshot saved: {screenshot_path}")
    
    source_image_path = None
    if source_image_data:
        source_image_path = os.path.join(output_dir, f"source_image_{timestamp}.png")
        with open(source_image_path, 'wb') as f:
            f.write(source_image_data)
        log(f"Source image saved: {source_image_path}")
    
    return {
        "status": "success",
        "message": "Images received and saved successfully",
        "screenshot_path": screenshot_path,
        "source_image_path": source_image_path,
    }


async def handle_client(websocket):
    """Handle a client connection."""
    global connection_counter
    connection_counter += 1
    conn_id = connection_counter
    
    client_addr = websocket.remote_address
    log(f"=== CONNECTION #{conn_id} OPENED ===")
    log(f"  Client address: {client_addr}")
    
    # Send welcome message
    try:
        welcome_msg = json.dumps({
            "type": "welcome",
            "message": "Connected to AutoLUT server",
            "connection_id": conn_id
        })
        await websocket.send(welcome_msg)
        log(f"  Sent welcome message to client #{conn_id}")
    except Exception as e:
        log(f"  ERROR sending welcome: {e}")
    
    message_count = 0
    try:
        async for message in websocket:
            message_count += 1
            log(f"--- Message #{message_count} from connection #{conn_id} ---")
            log(f"  Raw message length: {len(message)} bytes")
            
            try:
                data = json.loads(message)
                command = data.get("command", "")
                log(f"  Command: {command}")
                log(f"  Keys in message: {list(data.keys())}")
                
                if command == "ping":
                    response = {
                        "status": "success",
                        "type": "pong",
                        "message": "Connection verified",
                        "connection_id": conn_id
                    }
                    log(f"  Ping received, sending pong")
                
                elif command == "process_video":
                    # Check if video_data is present (new method: video sent as Base64)
                    video_b64 = data.get("video_data", "")
                    
                    log(f"  video_data present: {bool(video_b64)}")
                    log(f"  video_data length: {len(video_b64)} chars")
                    
                    if video_b64:
                        # New method: decode Base64 video data
                        filename = data.get("filename", "video.mp4")
                        total_frames = data.get("total_frames", 0)
                        fps = data.get("fps", 30)
                        
                        log(f"  Decoding video data ({len(video_b64)} chars Base64)...")
                        try:
                            video_data = base64.b64decode(video_b64)
                            log(f"  Decoded video size: {len(video_data)} bytes")
                            response = await process_video(video_data, filename, total_frames, fps)
                        except Exception as decode_err:
                            log(f"  ERROR decoding Base64: {decode_err}")
                            response = {
                                "status": "error",
                                "message": f"Failed to decode video data: {str(decode_err)}"
                            }
                    else:
                        # Legacy method: frame_sequence_path (not used anymore)
                        response = {
                            "status": "error",
                            "message": "No video_data provided. Please use the new video transfer method."
                        }
                
                elif command == "generate_lut":
                    # Legacy image-based method
                    screenshot_b64 = data.get("screenshot", "")
                    source_image_b64 = data.get("source_image", "")
                    
                    if not screenshot_b64:
                        response = {
                            "status": "error",
                            "message": "No screenshot data provided"
                        }
                    else:
                        screenshot_data = base64.b64decode(screenshot_b64)
                        source_image_data = base64.b64decode(source_image_b64) if source_image_b64 else None
                        response = await process_images(screenshot_data, source_image_data)
                
                else:
                    response = {
                        "status": "error",
                        "message": f"Unknown command: {command}"
                    }
                    log(f"  ERROR: Unknown command '{command}'")
                
                # Send response
                response_json = json.dumps(response)
                await websocket.send(response_json)
                log(f"  Response sent: {response.get('status', 'unknown')}")
                
            except json.JSONDecodeError as e:
                log(f"  ERROR: Invalid JSON - {e}")
                error_response = {"status": "error", "message": f"Invalid JSON: {str(e)}"}
                await websocket.send(json.dumps(error_response))
                
            except Exception as e:
                log(f"  ERROR processing message: {e}")
                log(f"  Traceback: {traceback.format_exc()}")
                error_response = {"status": "error", "message": f"Processing error: {str(e)}"}
                await websocket.send(json.dumps(error_response))
                
    except websockets.exceptions.ConnectionClosedOK:
        log(f"=== CONNECTION #{conn_id} CLOSED (OK) ===")
        log(f"  Messages received: {message_count}")
    except websockets.exceptions.ConnectionClosedError as e:
        log(f"=== CONNECTION #{conn_id} CLOSED (ERROR) ===")
        log(f"  Error: {e}")
    except Exception as e:
        log(f"=== CONNECTION #{conn_id} ERROR ===")
        log(f"  Error: {e}")
        log(f"  Traceback: {traceback.format_exc()}")


async def main():
    """Start the WebSocket server."""
    print("=" * 60)
    print("AutoLUT WebSocket Server")
    print("=" * 60)
    log(f"Server URL: ws://{HOST}:{PORT}")
    log(f"Output directory: {OUTPUT_DIR}")
    
    # Verify output directory is writable
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    test_file = os.path.join(OUTPUT_DIR, ".write_test")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        log(f"Output directory is writable: OK")
    except Exception as e:
        log(f"WARNING: Cannot write to output directory: {e}")
    
    print("-" * 60)
    log("Server started. Waiting for connections...")
    log("Videos will be received as Base64-encoded data from UE5 plugin")
    log("Press Ctrl+C to stop")
    print("-" * 60)
    
    async with websockets.serve(
        handle_client, 
        HOST, 
        PORT,
        ping_interval=20,
        ping_timeout=30,
        max_size=500 * 1024 * 1024,  # 500MB max message size for video data
    ):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Server stopped by user")
        print("=" * 60)
