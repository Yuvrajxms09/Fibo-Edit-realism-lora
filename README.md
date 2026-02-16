# Fibo Edit on Modal

Image editing with Bria’s [Fibo-Edit](https://huggingface.co/briaai/Fibo-Edit) on [Modal](https://modal.com): natural-language or VGL JSON prompts, the [character-consistency LoRA](https://huggingface.co/briaai/fibo_edit_character_consistency_lora) is loaded and fused into the model at startup and a local [FIBO VLM](https://huggingface.co/briaai/FIBO-vlm) for prompt→JSON . The web UI is in `frontend/`; models weights are staged once via `upload.py` into a shared Modal volume.

## Example outputs

Single input, three edits (same image, different prompts):

| Input | Output 1 | Output 2 | Output 3 |
|:---:|:---:|:---:|:---:|
| <img src="input.jpg" width="280"/> | <img src="output.png" width="280"/> | <img src="output2.png" width="280"/> | <img src="output3.png" width="280"/> |
| — | *summer chic dress* | *red stylish gown* | *emerald green satin evening gown* |

**Prompts:** (1) *change the outfit to a summer chic dress* — (2) *put her in a red stylish gown* — (3) *Replace clothing with a fitted emerald green satin evening gown, subtle jewelry, soft curls. Cinematic lighting, luxury fashion editorial, shallow depth of field.*

*Input image: [Tatyana Doloman](https://www.pexels.com/@tatyana-doloman-728740365/) on Pexels (free to use).*

## Files Inculded

- **`inference.py`** — Modal app:  Loads the fibo edit model and lora from the volume, then fuses the LoRA into the transformer so that the running model already has it. It also serves the frontend and the REST API.
- **`upload.py`** — run it once at start: downloads weights of Fibo-Edit, FIBO-vlm, and the character-consistency LoRA from Hugging Face into the Modal volume `fibo-edit-assets`. It only populates the volume that `inference.py` mounts.
- **`frontend/`** — Single-page UI: upload image or paste URL, type an edit instruction, get the edited image (async flow: POST then poll for result).


## Pipeline behaviour

1. **Fibo-Edit** expects a VGL-style JSON prompt. If the user sends plain text, that text is first converted to JSON.
2. **Prompt→JSON** is done on the GPU worker with the local [FIBO-vlm](https://huggingface.co/briaai/FIBO-vlm) (Fibo-Edit’s local VLM path). No external VLM or API key.
3. **LoRA** — On startup, `inference.py` loads the character-consistency LoRA from `/models/fibo_edit_character_consistency_lora` with PEFT and calls `merge_and_unload()`, so the LoRA is fused into the transformer. At inference time there’s no separate LoRA; it’s part of the model. (If that path is missing from the volume, the app runs base model only.)
4. All heavy assets live in the Modal volume and are mounted at `/models` in the worker.

## Prerequisites

- Modal account and CLI: `pip install modal` then `modal token new`.
- Hugging Face token (for gated models). Create a Modal secret named `huggingface-secret` with `HF_TOKEN=<your_token>`.

## Steps

### 0. Clone required repos

`inference.py` bundles Fibo-Edit source and the FIBO VLM prompt-to-JSON pipeline into the Modal image via local paths. From the project root, clone both repos:

```bash
git clone https://huggingface.co/briaai/FIBO-VLM-prompt-to-JSON
git clone https://github.com/Bria-AI/Fibo-Edit.git
```


### 1. Populate the volume (once)

From the project root:

```bash
modal run upload.py
```

This runs the `upload_models_to_volume` logic in `upload.py`: it downloads the model weights into `fibo-edit-assets`.

### 2. Deploy the app

```bash
modal deploy inference.py
```

Modal prints the web app URL (e.g. `https://<workspace>--fibo-edit-web-endpoint.modal.run`). Open it in a browser.

### 3. Using the UI

At the app root you get the frontend. Provide an image (upload or URL), enter a prompt and click edit, the backend converts the prompt to JSON with the local FIBO VLM, runs Fibo-Edit (LoRA already fused in) and returns the edited image. No Gemini or other external API is needed/used.

## API

Same app exposes REST endpoints:

- **POST `/v1/images/edit`** — Synchronous edit. Form fields: `prompt`, and either `image` (file) or `image_url`. Optional: `num_inference_steps`, `guidance_scale`, `seed`, `negative_prompt`, `num_images_per_prompt`, `max_sequence_length`, `do_patching`. Query/header `return_format`: `json` (base64 in body) or `binary` (raw PNG).
- **POST `/v1/images/edit/async`** — Async edit. Same form; returns `call_id`. Poll **GET `/v1/images/edit/result/{call_id}`** until you get the result (202 while still running).
- **POST `/v1/images/edit/json`** — JSON body: `prompt`, and either `image_base64` or `image_url`, plus optional pipeline params.

The frontend uses the async flow: POST to `/v1/images/edit/async`, then poll `/v1/images/edit/result/{call_id}`.

## Summary

| Piece | Role |
|-------|------|
| **inference.py** | Modal app: Fibo-Edit + character-consistency LoRA fused at startup + local FIBO-vlm (prompt→JSON); serves frontend and REST API. |
| **upload.py** | Fills volume `fibo-edit-assets` with [Fibo-Edit](https://huggingface.co/briaai/Fibo-Edit), [FIBO-vlm](https://huggingface.co/briaai/FIBO-vlm), and [fibo_edit_character_consistency_lora](https://huggingface.co/briaai/fibo_edit_character_consistency_lora). |
| **frontend/** | Web UI: image + natural-language edit; calls the deployed app. |

Run `modal run upload.py` once to populate the volume, then `modal deploy inference.py` and open the given URL.
