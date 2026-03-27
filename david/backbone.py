"""Frozen Qwen3-VL vision backbone for extracting video visual features."""

import gc

import torch
from torch import Tensor
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from .utils import pad_sequence_to_max


class Qwen3VLBackbone:
    """Wraps the Qwen3-VL vision encoder (frozen) and exposes a clean feature extraction API.

    Loads the full model onto CPU, extracts only the vision encoder
    (model.model.visual) onto the target device, then frees the LLM weights.
    This saves ~16 GB of VRAM compared to keeping the full 8B model on GPU.

    Output features come from pooler_output (post-PatchMerger, dim=4096 for 8B model),
    which is the representation the LLM actually consumes.
    """

    def __init__(self, model_name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.device = device
        self.dtype = dtype

        print(f"Loading Qwen3-VL backbone: {model_name}")
        # Load full model onto CPU first, then extract only the vision encoder.
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cpu",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Move only the vision encoder to the target device; discard the LLM.
        self._visual = model.model.visual.to(device)
        self._visual.eval()
        for param in self._visual.parameters():
            param.requires_grad_(False)

        del model
        gc.collect()
        if "cuda" in device:
            torch.cuda.empty_cache()

        print("Backbone loaded and frozen.")

    @torch.no_grad()
    def extract_features(
        self,
        pixel_values: Tensor,
        grid_thw: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Extract visual features from preprocessed video frames.

        Args:
            pixel_values: Preprocessed patch tokens — shape depends on the processor
                          but is typically [total_patches, C*patch_t*patch_h*patch_w].
            grid_thw: Grid dimensions per video — [num_videos, 3] where each row
                      is [T_grid, H_grid, W_grid].

        Returns:
            features: [batch, N_max, D] float32 padded feature tensor.
            mask:     [batch, N_max] bool mask — True = valid token.
        """
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)
        grid_thw = grid_thw.to(self.device)
        # HF Qwen3-VL vision API changed across versions:
        # - newer: ModelOutput with `pooler_output`
        # - older: tuple-like outputs
        try:
            vision_out = self._visual(pixel_values, grid_thw, return_dict=True)
            pooler = vision_out.pooler_output
            if pooler is None:
                raise AttributeError("vision_out.pooler_output is None")
        except (TypeError, AttributeError):
            pooler, _deepstack = self._visual(pixel_values, grid_thw)

        # Each video produces T * (H//2) * (W//2) tokens after spatial merge (factor=2)
        N_per_video = [
            int(t) * int(h // 2) * int(w // 2)
            for t, h, w in grid_thw.tolist()
        ]

        # Split flat tensor back into per-video tensors
        features_list = list(pooler.split(N_per_video, dim=0))

        # Cast to float32 for stable VAE training
        features_list = [f.float() for f in features_list]

        # Pad to uniform length
        features, mask = pad_sequence_to_max(features_list)
        features = features.to(self.device)
        mask = mask.to(self.device)

        return features, mask

    def extract_features_from_frames(
        self,
        frames,
        sample_fps: float,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run a numpy frame array through the processor and vision backbone.

        Args:
            frames: [T, H, W, C] numpy array of video frames.
            sample_fps: Frame rate used when sampling (for per-frame timestamps).
            min_pixels: Optional lower bound on pixel budget for resizing.
            max_pixels: Optional upper bound on pixel budget for resizing.

        Returns:
            features: [1, N, D]
            mask:     [1, N]
        """
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        frame_list = [Image.fromarray(f) for f in frames]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video", 
                        "video": frame_list, 
                        "fps": sample_fps,
                        # "nframes": len(frame_list),
                    },
                    {"type": "text", "text": "Describe the video."},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        # process_vision_info cannot infer fps from PIL images; provide VideoMetadata
        # explicitly so Qwen3-VL builds correct per-frame timestamps.
        video_kwargs.pop("fps", None)

        processor_kwargs = dict(
            text=[text],
            videos=video_inputs,
            video_metadata=[{
                "fps": sample_fps,
                "total_num_frames": len(frame_list),
                "frames_indices": list(range(len(frame_list))),
            }],
            **video_kwargs,
            return_tensors="pt",
        )
        if min_pixels is not None or max_pixels is not None:
            vk = {}
            if min_pixels is not None:
                vk["min_pixels"] = min_pixels
            if max_pixels is not None:
                vk["max_pixels"] = max_pixels
            processor_kwargs["videos_kwargs"] = vk

        inputs = self.processor(**processor_kwargs)
        return self.extract_features(inputs["pixel_values_videos"], inputs["video_grid_thw"])

    def get_processor(self):
        return self.processor
