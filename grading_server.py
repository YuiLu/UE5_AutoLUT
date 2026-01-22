#!/usr/bin/env python3
"""
VideoColorGrading WebSocket Server
This script receives image/video data from clients and performs color grading inference.

Requirements:
    pip install websockets

Usage:
    python grading_server.py

The server listens on ws://127.0.0.1:8766 by default.
"""

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch\.utils\._pytree\._register_pytree_node` is deprecated\.",
)

import asyncio
import websockets
import json
import base64
import os
import io
import traceback
import inspect
import numpy as np
from datetime import datetime
from PIL import Image, ImageFilter

from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from tqdm import tqdm
from transformers import CLIPProcessor

from models.ImageEncoder import ImageEncoder
from models.ReferenceNet import ReferenceNet
from pipeline import InferencePipeline
from diffusers.models import UNet2DConditionModel

from utils.util import save_videos_grid, preprocess, save_cube_lut, generate_combined_lut
from utils.videoreader import VideoReader

from accelerate.utils import set_seed
from einops import rearrange
from pillow_lut import identity_table

# Configuration
HOST = "127.0.0.1"
PORT = 8766
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_output")
CONFIG_PATH = "configs/prompts/video_demo.yaml"

# Global inference instance
grader = None
connection_counter = 0


def log(msg: str):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}")


class InferenceServer:
    """
    Inference server that accepts raw bytes instead of file paths.
    """
    def __init__(self, config="configs/prompts/video_demo.yaml") -> None:
        print("Initializing LUT Generation Pipeline...")
        config = OmegaConf.load(config)
        inference_config = OmegaConf.load(config.inference_config)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create diffusion pipeline
        vae = AutoencoderKL.from_pretrained(config.pretrained_sd_path, subfolder="vae")
        self.clip_image_encoder = ImageEncoder(model_path=config.pretrained_clip_path)
        self.clip_image_processor = CLIPProcessor.from_pretrained(config.pretrained_clip_path, local_files_only=True)

        unet = UNet2DConditionModel.from_pretrained(
            config.pretrained_sd_path, subfolder="unet",
            in_channels=6, out_channels=3,
            low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        )
        state_dict = torch.load(config.pretrained_LD_path, map_location="cpu")['unet_state_dict']
        m, u = unet.load_state_dict(state_dict, strict=True)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)} ###")
        if len(m) != 0 or len(u) != 0:
            print(f"### missing keys:\n{m}\n### unexpected keys:\n{u}\n ###")
        
        self.referencenet = ReferenceNet.from_pretrained(config.pretrained_sd_path, subfolder="unet")
        state_dict = torch.load(config.pretrained_GE_path, map_location="cpu")["referencenet_state_dict"]
        m, u = self.referencenet.load_state_dict(state_dict, strict=True)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)} ###")
        if len(m) != 0 or len(u) != 0:
            print(f"### missing keys:\n{m}\n### unexpected keys:\n{u}\n ###")

        self.id_lut_hwc = identity_table(16).table.reshape(64, 64, 3)
        self.id_lut_chw = torch.from_numpy(rearrange(self.id_lut_hwc, "h w c -> c h w")).unsqueeze(0)

        self.pipeline = InferencePipeline(
            vae=vae, unet=unet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        )
        self.pipeline.to(device)
        print("Initialization Done!")

    def run_inference(
        self,
        ref_image_data: bytes,
        input_video_data: bytes,
        save_path: str,
        random_seed: int = 48,
        step: int = 25,
        size: int = 512,
        ncc: bool = False
    ) -> str:
        """
        Run color grading inference from raw bytes.
        
        Args:
            ref_image_data: Reference image bytes (PNG/JPG)
            input_video_data: Input video bytes (MP4/etc)
            save_path: Path to save output video
            random_seed: Random seed for inference
            step: Number of inference steps
            size: Image size for processing
            ncc: Enable color correction
            
        Returns:
            Path to saved output video
        """
        # Load reference image from bytes
        ref_image = Image.open(io.BytesIO(ref_image_data)).convert('RGB')
        ref_image = ref_image.resize((size, size))
        ref_sequence = np.array(ref_image)
        
        # Load input video from bytes using VideoReader
        # VideoReader uses av.open() which accepts file-like objects
        input_video = VideoReader(io.BytesIO(input_video_data)).read()
        input_video, input_video_resize, preprocess_params = preprocess(input_video, ref_sequence, size, ncc)

        random_seed = int(random_seed)
        if random_seed != -1:
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()
        step = int(step)

        generator = torch.Generator(device=torch.device("cuda:0"))
        generator.manual_seed(torch.initial_seed())
    
        lut = self.pipeline(
            num_inference_steps=step,
            width=size,
            height=size,
            generator=generator,
            num_actual_inference_steps=step,
            source_image=ref_sequence,
            referencenet=self.referencenet,
            clip_image_processor=self.clip_image_processor,
            clip_image_encoder=self.clip_image_encoder,
            input_video=input_video_resize,
            id_lut=self.id_lut_chw,
            return_dict=False
        )
    
        lut = lut[0].detach().cpu().numpy()
        lut = rearrange(lut, "c h w -> h w c")
        lut_full = lut + self.id_lut_hwc
        lut_full = np.clip(lut_full, 0.0, 1.0)

        # Save .cube LUT file
        # Generate combined LUT that includes preprocess transform for external software
        base, _ext = os.path.splitext(save_path)
        cube_path = base + ".cube"
        combined_lut = generate_combined_lut(lut_full, preprocess_params, lut_size=16)
        save_cube_lut(combined_lut, cube_path, title=os.path.basename(base), lut_size=16)

        lut_filter = ImageFilter.Color3DLUT(16, lut_full.flatten())
        output_frames = []

        for frame in tqdm(input_video):
            output_frame = np.array(Image.fromarray(frame).filter(lut_filter)) / 255.0
            output_frames.append(output_frame)
        output_frames = np.array(output_frames)
        output_frames = rearrange(output_frames, "t h w c -> 1 c t h w")
        output_video = torch.from_numpy(output_frames)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_videos_grid(output_video, save_path)
        
        return save_path, cube_path, lut_full


def init_grader():
    """Initialize the inference model."""
    global grader
    if grader is None:
        log("Loading inference model...")
        grader = InferenceServer(config=CONFIG_PATH)
        log("Model loaded successfully!")
    return grader


async def process_color_grading(
    ref_image_data: bytes,
    input_video_data: bytes,
    seed: int = 48,
    steps: int = 25,
    size: int = 512,
    ncc: bool = False
) -> dict:
    """
    Process color grading with reference image and input video.
    
    Args:
        ref_image_data: Reference image bytes (for color style)
        input_video_data: Input video bytes (to be graded)
        seed: Random seed for inference
        steps: Number of inference steps
        size: Image size for processing
        ncc: Enable color correction
    
    Returns:
        dict with status and output data
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    output_path = os.path.join(OUTPUT_DIR, f"output_{timestamp}.mp4")
    
    log(f"Received reference image: {len(ref_image_data)} bytes")
    log(f"Received input video: {len(input_video_data)} bytes")
    
    # Run inference
    log("Starting color grading inference...")
    try:
        result_path, cube_path, lut_full = grader.run_inference(
            ref_image_data=ref_image_data,
            input_video_data=input_video_data,
            save_path=output_path,
            random_seed=seed,
            step=steps,
            size=size,
            ncc=ncc
        )
        log(f"Inference complete! Output: {result_path}")
        
        # Read the generated .cube LUT file
        lut_data = None
        if os.path.exists(cube_path):
            with open(cube_path, 'rb') as f:
                lut_data = base64.b64encode(f.read()).decode('utf-8')
            log(f"LUT file loaded: {cube_path}")
        
        # Note: Not returning output_video_data to avoid exceeding WebSocket message size limits
        # The output video is saved at result_path if needed
        
        return {
            "status": "success",
            "message": "Color grading completed successfully",
            "output_video_path": result_path,
            "output_lut_path": cube_path,
            "lut_data": lut_data,
            "timestamp": timestamp
        }
        
    except Exception as e:
        log(f"Inference error: {e}")
        log(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Inference failed: {str(e)}",
            "timestamp": timestamp
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
            "message": "Connected to VideoColorGrading server",
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
                
                if command == "ping":
                    response = {
                        "status": "success",
                        "type": "pong",
                        "message": "Connection verified",
                        "connection_id": conn_id
                    }
                    log(f"  Ping received, sending pong")
                    
                elif command == "color_grading":
                    # Get data (base64 encoded)
                    ref_image_b64 = data.get("ref_image", "")
                    input_video_b64 = data.get("input_video", "")
                    
                    # Get optional parameters
                    seed = data.get("seed", 48)
                    steps = data.get("steps", 25)
                    size = data.get("size", 512)
                    ncc = data.get("ncc", False)
                    
                    log(f"  Reference image base64 length: {len(ref_image_b64)}")
                    log(f"  Input video base64 length: {len(input_video_b64)}")
                    log(f"  Parameters: seed={seed}, steps={steps}, size={size}, ncc={ncc}")
                    
                    if not ref_image_b64 or not input_video_b64:
                        response = {
                            "status": "error",
                            "message": "Both ref_image and input_video are required"
                        }
                        log(f"  ERROR: Missing data")
                    else:
                        # Decode data
                        log(f"  Decoding base64 data...")
                        ref_image_data = base64.b64decode(ref_image_b64)
                        input_video_data = base64.b64decode(input_video_b64)
                        
                        log(f"  Decoded reference image: {len(ref_image_data)} bytes")
                        log(f"  Decoded input video: {len(input_video_data)} bytes")
                        
                        # Process color grading
                        log(f"  Processing color grading...")
                        response = await process_color_grading(
                            ref_image_data=ref_image_data,
                            input_video_data=input_video_data,
                            seed=seed,
                            steps=steps,
                            size=size,
                            ncc=ncc
                        )
                        log(f"  Processing complete!")
                        
                elif command == "get_status":
                    response = {
                        "status": "success",
                        "type": "status",
                        "model_loaded": grader is not None,
                        "connection_id": conn_id
                    }
                    
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
                error_response = {
                    "status": "error",
                    "message": f"Invalid JSON: {str(e)}"
                }
                await websocket.send(json.dumps(error_response))
                
            except Exception as e:
                log(f"  ERROR processing message: {e}")
                log(f"  Traceback: {traceback.format_exc()}")
                error_response = {
                    "status": "error",
                    "message": f"Processing error: {str(e)}"
                }
                await websocket.send(json.dumps(error_response))
                
    except websockets.exceptions.ConnectionClosedOK:
        log(f"=== CONNECTION #{conn_id} CLOSED (OK) ===")
        log(f"  Messages received: {message_count}")
    except websockets.exceptions.ConnectionClosedError as e:
        log(f"=== CONNECTION #{conn_id} CLOSED (ERROR) ===")
        log(f"  Error: {e}")
        log(f"  Messages received: {message_count}")
    except Exception as e:
        log(f"=== CONNECTION #{conn_id} ERROR ===")
        log(f"  Error: {e}")
        log(f"  Traceback: {traceback.format_exc()}")


async def main():
    """Start the WebSocket server."""
    print("=" * 60)
    print("VideoColorGrading WebSocket Server")
    print("=" * 60)
    
    # Initialize model at startup
    init_grader()
    
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
    log("Press Ctrl+C to stop")
    print("-" * 60)
    print("\nAPI Commands:")
    print("  - ping: Test connection")
    print("  - get_status: Get server status")
    print("  - color_grading: Run inference")
    print("    Required: ref_image (base64), input_video (base64)")
    print("    Optional: seed (int), steps (int), size (int), ncc (bool)")
    print("-" * 60)
    
    async with websockets.serve(
        handle_client,
        HOST,
        PORT,
        ping_interval=20,
        ping_timeout=30,
        max_size=500 * 1024 * 1024,  # 500MB max message size for videos
    ):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Server stopped by user")
        print("=" * 60)
