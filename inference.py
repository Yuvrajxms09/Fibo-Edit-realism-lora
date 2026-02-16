import os
import uuid
import tempfile
import modal
import base64
import json
from typing import Optional

app = modal.App("fibo")

volume = modal.Volume.from_name("fibo-edit-assets", create_if_missing=False)

MODEL_PATH = "/models"
FIBO_EDIT_SRC = "/opt/fibo_edit_src"
VLM_PIPELINE_TEMPLATE = "/opt/vlm_pipeline_template"

fibo_edit_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "git-lfs")
    .run_commands("git lfs install")
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), "Fibo-Edit", "src"),
        remote_path=FIBO_EDIT_SRC,
        copy=True,
    )
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), "FIBO-VLM-prompt-to-JSON"),
        remote_path=VLM_PIPELINE_TEMPLATE,
        copy=True,
    )
    .uv_pip_install(
        "boltons>=25.0.0",
        "litellm>=1.80.16",
        "pillow>=12.1.0",
        "pydantic>=2.12.5",
        "requests>=2.32.5",
        "ujson>=5.11.0",
        "accelerate",
        "protobuf>=6.33.4",
        "sentencepiece",
        "transformers",
        "torch",
        "torchvision",
        "datasets>=4.5.0",
        "peft>=0.18.1",
        "numpy",
        "scipy",
        "huggingface_hub",
        extra_options="--index-strategy unsafe-best-match",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands("pip install wheel setuptools")
    .run_commands("pip install git+https://github.com/huggingface/diffusers.git")
)

frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "pydantic==2.12.5",
        "python-multipart==0.0.6",
        "httpx>=0.24.0",
    )
    .add_local_dir(frontend_path, remote_path="/assets")
)


@app.cls(image=fibo_edit_image, gpu="L40S", scaledown_window=180, timeout=900, volumes={MODEL_PATH: volume})
@modal.concurrent(max_inputs=4, target_inputs=3)
class FiboEditWorker:
    """Single promptify path: Fibo-Edit get_prompt (vlm_mode=local) → generate_prompt_local → ModularPipelineBlocks + FIBO-vlm from volume."""
    _is_loaded = False
    _current_lora: Optional[str] = None  # Fused LoRA name, or None if base only

    def _promptify(self, image, instruction: str) -> str:
        import sys
        import gc
        import shutil
        import torch

        print("[PROMPTIFY] Using single path: Fibo-Edit get_prompt (vlm_mode=local) with FIBO-vlm from volume")
        vlm_path = os.path.join(MODEL_PATH, "FIBO-vlm")
        if not os.path.exists(vlm_path):
            print(f"[PROMPTIFY] ❌ FIBO-vlm not found at {vlm_path}")
            raise FileNotFoundError(f"FIBO-vlm not found in volume at {vlm_path}. Upload it first.")

        if FIBO_EDIT_SRC not in sys.path:
            sys.path.insert(0, FIBO_EDIT_SRC)

        config_dir = "/tmp/fibo_vlm_local_config"
        os.makedirs(config_dir, exist_ok=True)
        for f in os.listdir(VLM_PIPELINE_TEMPLATE):
            src = os.path.join(VLM_PIPELINE_TEMPLATE, f)
            dst = os.path.join(config_dir, f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif not os.path.exists(dst):
                shutil.copytree(src, dst)

        config_path = os.path.join(config_dir, "config.json")
        with open(config_path, "r") as f:
            cfg = json.load(f)
        cfg["model_id"] = vlm_path
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[PROMPTIFY] Config ready: model_id={vlm_path}")

        try:
            from fibo_edit.edit_promptify import get_prompt

            print("[PROMPTIFY] Calling Fibo-Edit get_prompt (local VLM)...")
            out = get_prompt(
                image=image,
                instruction=instruction,
                model=config_dir,
                vlm_mode="local",
            )
            print(f"[PROMPTIFY] ✅ VLM returned JSON prompt (len={len(out)})")
            return out
        except Exception as e:
            print(f"[PROMPTIFY] ❌ VLM failed: {e}")
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    @modal.enter()
    def load_pipeline(self):
        import time
        import torch
        from diffusers import BriaFiboEditPipeline

        if self._is_loaded:
            return

        print("[GPU START] Loading Fibo Edit pipeline...")
        start_time = time.time()

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        local_model_path = os.path.join(MODEL_PATH, "Fibo-Edit")
        print(f"[GPU START] Loading model from volume: {local_model_path}")
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Model not found in volume at {local_model_path}. Please upload model files first.")
        print(f"[GPU START] Model path verified, files present: {os.listdir(local_model_path)[:5]}...")

        try:
            self.pipeline = BriaFiboEditPipeline.from_pretrained(
                local_model_path,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            self.pipeline.to("cuda")

            default_lora = "fibo_edit_character_consistency_lora"
            lora_path = os.path.join(MODEL_PATH, default_lora)
            if os.path.exists(lora_path):
                try:
                    print(f"[GPU START] Loading character consistency LoRA (PEFT format): {default_lora}")
                    from peft import PeftModel
                    self.pipeline.transformer = PeftModel.from_pretrained(
                        self.pipeline.transformer,
                        lora_path,
                    )
                    if hasattr(self.pipeline.transformer, "merge_and_unload"):
                        self.pipeline.transformer = self.pipeline.transformer.merge_and_unload()
                        print("[GPU START] ✅ Character consistency LoRA loaded and fused")
                    else:
                        print("[GPU START] ✅ Character consistency LoRA loaded")
                    self._current_lora = default_lora
                except Exception as e:
                    print(f"[GPU START] ⚠️  LoRA load failed: {e}, continuing with base only")
                    self._current_lora = None
            else:
                print(f"[GPU START] No LoRA at {lora_path}, base model only")
                self._current_lora = None

            load_time = time.time() - start_time
            print(f"[GPU START] Pipeline loaded in {load_time:.1f}s!")
            self._is_loaded = True
        except Exception as e:
            print(f"[GPU START] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def save_to_volume(self, image_path: str) -> str:
        import shutil
        
        output_dir = os.path.join(MODEL_PATH, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"fibo_edit_{uuid.uuid4().hex[:12]}.png"
        volume_path = os.path.join(output_dir, filename)
        shutil.copy2(image_path, volume_path)
        
        return volume_path

    def process_edit(
        self,
        image_data: bytes,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 3000,
        do_patching: bool = False,
    ) -> str:
        import time
        from PIL import Image
        import io
        import torch

        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required for image editing")

        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        prompt_str = prompt.strip()
        try:
            json.loads(prompt_str)
            formatted_prompt = prompt_str
            print(f"[GPU TASK] Prompt is valid JSON, using as-is (no VLM)")
        except (json.JSONDecodeError, ValueError):
            print(f"[GPU TASK] Prompt is natural language, running VLM promptify (Fibo-Edit local)")
            formatted_prompt = self._promptify(image, prompt_str)
            parsed = json.loads(formatted_prompt)
            if isinstance(parsed, dict) and "edit_instruction" not in parsed:
                parsed["edit_instruction"] = prompt_str
                formatted_prompt = json.dumps(parsed)

        print(f"[GPU TASK] Processing image edit with prompt: {prompt_str[:100]}...")
        start_time = time.time()

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        pipeline_kwargs = {
            "image": image,
            "prompt": formatted_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "max_sequence_length": max_sequence_length,
            "do_patching": do_patching,
        }
        if generator is not None:
            pipeline_kwargs["generator"] = generator
        
        if negative_prompt is not None:
            pipeline_kwargs["negative_prompt"] = negative_prompt.strip()

        result = self.pipeline(**pipeline_kwargs).images[0]

        # Save to temp file
        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, "edited_image.png")
        result.save(temp_image_path)

        elapsed = time.time() - start_time
        print(f"[GPU TASK] ✅ Image editing completed in {elapsed:.1f}s")
        return temp_image_path

    @modal.method()
    def generate_edit_and_save(
        self,
        image_data: bytes,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 3000,
        do_patching: bool = False,
    ) -> tuple:
        temp_image_path = self.process_edit(
            image_data,
            prompt,
            num_inference_steps,
            guidance_scale,
            seed,
            negative_prompt,
            num_images_per_prompt,
            max_sequence_length,
            do_patching,
        )
        
        with open(temp_image_path, 'rb') as f:
            image_data_result = f.read()
        
        volume_path = self.save_to_volume(temp_image_path)
        
        os.remove(temp_image_path)
        os.rmdir(os.path.dirname(temp_image_path))
        
        return volume_path, image_data_result

    @modal.method()
    def get_model_info(self) -> dict:
        return {
            "is_loaded": self._is_loaded,
            "model_name": "briaai/Fibo-Edit",
            "current_lora": self._current_lora,
        }


@app.function(image=web_image)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="fibo-edit-web-endpoint")
def fastapi_app():
    from fastapi import FastAPI, Form, UploadFile, File
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel, Field, ConfigDict
    from typing import Optional
    import httpx
    
    class EditRequest(BaseModel):
        model_config = ConfigDict(protected_namespaces=())
        prompt: str = Field(..., description="Structured JSON prompt (as string) or edit instruction")
        image_url: Optional[str] = Field(default=None, description="Input image URL to edit")
        image_base64: Optional[str] = Field(default=None, description="Input image as base64 string (alternative to image_url)")
        num_inference_steps: Optional[int] = Field(default=50, ge=1, le=100, description="Number of inference steps")
        guidance_scale: Optional[float] = Field(default=3.5, ge=1.0, le=20.0, description="Guidance scale")
        seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
        negative_prompt: Optional[str] = Field(default=None, description="Negative prompt to guide generation away from")
        num_images_per_prompt: Optional[int] = Field(default=1, ge=1, le=4, description="Number of images to generate per prompt")
        max_sequence_length: Optional[int] = Field(default=3000, ge=1, le=3000, description="Maximum sequence length for prompt")
        do_patching: Optional[bool] = Field(default=False, description="Whether to use patching mode")

    async def resolve_image_bytes(
        image_file: Optional[UploadFile] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
    ) -> tuple[Optional[bytes], Optional[JSONResponse]]:
        if not image_file and not image_url and not image_base64:
            return None, JSONResponse(status_code=400, content={
                "status": "error",
                "message": "Provide 'image' file, 'image_url', or 'image_base64'",
            })
        if image_file:
            return await image_file.read(), None
        if image_base64:
            try:
                s = image_base64.split(",", 1)[1] if image_base64.startswith("data:image") else image_base64
                return base64.b64decode(s), None
            except Exception as e:
                return None, JSONResponse(status_code=400, content={
                    "status": "error",
                    "message": f"Invalid base64 image: {e}",
                })
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(image_url)
                response.raise_for_status()
                return response.content, None
        except Exception as e:
            return None, JSONResponse(status_code=400, content={
                "status": "error",
                "message": f"Failed to fetch image from URL: {e}",
            })

    web_app = FastAPI(title="Fibo Edit API", version="1.0.0")

    @web_app.get("/")
    async def serve_frontend():
        from fastapi.responses import FileResponse
        return FileResponse("/assets/index.html")

    @web_app.get("/favicon.ico")
    async def favicon():
        from fastapi.responses import Response
        return Response(status_code=204)

    @web_app.post("/v1/images/edit/async")
    async def edit_image_async(
        prompt: str = Form(..., description="Edit instruction or structured JSON prompt"),
        image: Optional[UploadFile] = File(None),
        image_url: Optional[str] = Form(None),
        num_inference_steps: int = Form(50, ge=1, le=100),
        guidance_scale: float = Form(3.5, ge=1.0, le=20.0),
        seed: Optional[int] = Form(None),
        negative_prompt: Optional[str] = Form(None),
        num_images_per_prompt: int = Form(1, ge=1, le=4),
        max_sequence_length: int = Form(3000, ge=1, le=3000),
        do_patching: bool = Form(False),
    ):
        image_data, err = await resolve_image_bytes(image_file=image, image_url=image_url)
        if err:
            return err
        call = FiboEditWorker().generate_edit_and_save.spawn(
            image_data=image_data,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            do_patching=do_patching,
        )
        return JSONResponse(content={"call_id": call.object_id, "status": "accepted"})

    @web_app.get("/v1/images/edit/result/{call_id}")
    async def edit_image_result(call_id: str):
        try:
            function_call = modal.FunctionCall.from_id(call_id)
            volume_path, result_image_data = function_call.get(timeout=0)
        except TimeoutError:
            return Response(status_code=202)
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
        image_base64 = base64.b64encode(result_image_data).decode("utf-8")
        return JSONResponse(content={
            "status": "success",
            "image": f"data:image/png;base64,{image_base64}",
            "image_base64": image_base64,
            "volume_path": volume_path,
        })

    @web_app.post("/v1/images/edit")
    async def edit_image_endpoint(
        prompt: str = Form(..., description="Structured JSON prompt (as string) or edit instruction"),
        image: Optional[UploadFile] = File(None, description="Input image file to edit"),
        image_url: Optional[str] = Form(None, description="Input image URL to edit (alternative to image file)"),
        num_inference_steps: int = Form(50, ge=1, le=100),
        guidance_scale: float = Form(3.5, ge=1.0, le=20.0),
        seed: Optional[int] = Form(None),
        negative_prompt: Optional[str] = Form(None, description="Negative prompt to guide generation away from"),
        num_images_per_prompt: int = Form(1, ge=1, le=4),
        max_sequence_length: int = Form(3000, ge=1, le=3000),
        do_patching: bool = Form(False),
        return_format: str = Form("json", description="Response format: 'json' (base64 image) or 'binary' (raw image)"),
    ):
        try:
            image_data, err = await resolve_image_bytes(image_file=image, image_url=image_url)
            if err:
                return err

            volume_path, result_image_data = FiboEditWorker().generate_edit_and_save.remote(
                image_data=image_data,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                do_patching=do_patching,
            )
            
            # Return JSON response with base64 encoded image
            if return_format.lower() == "json":
                image_base64 = base64.b64encode(result_image_data).decode("utf-8")
                return JSONResponse(
                    content={
                        "status": "success",
                        "image": f"data:image/png;base64,{image_base64}",
                        "image_base64": image_base64,
                        "volume_path": volume_path,
                        "metadata": {
                            "prompt": prompt,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                            "seed": seed,
                            "num_images_per_prompt": num_images_per_prompt,
                            "max_sequence_length": max_sequence_length,
                            "do_patching": do_patching,
                        }
                    }
                )
            else:
                # Return binary image response
                return Response(
                    content=result_image_data,
                    media_type="image/png",
                    headers={
                        "Content-Disposition": 'attachment; filename="edited_image.png"',
                        "X-Modal-Volume-Path": volume_path
                    }
                )
        except Exception as e:
            error_msg = str(e)
            status_code = 400 if error_msg.startswith("400:") else 500
            message = error_msg[4:].strip() if error_msg.startswith(("400:", "500:")) else f"Error: {error_msg}"
            
            return JSONResponse(
                status_code=status_code,
                content={"status": "error", "message": message}
            )

    @web_app.post("/v1/images/edit/json")
    async def edit_image_endpoint_json(request: EditRequest):
        try:
            image_data, err = await resolve_image_bytes(
                image_url=request.image_url,
                image_base64=request.image_base64,
            )
            if err:
                return err

            volume_path, result_image_data = FiboEditWorker().generate_edit_and_save.remote(
                image_data=image_data,
                prompt=request.prompt,
                num_inference_steps=request.num_inference_steps or 50,
                guidance_scale=request.guidance_scale or 3.5,
                seed=request.seed,
                negative_prompt=request.negative_prompt,
                num_images_per_prompt=request.num_images_per_prompt or 1,
                max_sequence_length=request.max_sequence_length or 3000,
                do_patching=request.do_patching or False,
            )
            
            # Always return JSON response with base64 encoded image
            image_base64 = base64.b64encode(result_image_data).decode("utf-8")
            return JSONResponse(
                content={
                    "status": "success",
                    "image": f"data:image/png;base64,{image_base64}",
                    "image_base64": image_base64,
                    "volume_path": volume_path,
                    "metadata": {
                        "prompt": request.prompt,
                        "num_inference_steps": request.num_inference_steps or 50,
                        "guidance_scale": request.guidance_scale or 3.5,
                        "seed": request.seed,
                        "num_images_per_prompt": request.num_images_per_prompt or 1,
                        "max_sequence_length": request.max_sequence_length or 3000,
                        "do_patching": request.do_patching or False,
                    }
                }
            )
        except Exception as e:
            error_msg = str(e)
            status_code = 400 if error_msg.startswith("400:") else 500
            message = error_msg[4:].strip() if error_msg.startswith(("400:", "500:")) else f"Error: {error_msg}"
            
            return JSONResponse(
                status_code=status_code,
                content={"status": "error", "message": message}
            )

    @web_app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "fibo-edit"}

    @web_app.get("/model-info")
    async def get_model_info():
        try:
            return await FiboEditWorker().get_model_info.remote()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return web_app