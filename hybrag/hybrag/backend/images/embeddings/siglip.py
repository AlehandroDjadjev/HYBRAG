from __future__ import annotations
from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image
import io
from typing import Union, List

class SiglipService:
	def __init__(self, model_name: str | None = None, device: str | None = None):
		# Exact model per spec
		self.model_name = model_name or 'google/siglip2-giant-opt-patch16-384'
		# Load model and processor; device_map='auto' for local GPU/CPU placement
		self.model = AutoModel.from_pretrained(self.model_name, device_map='auto').eval()
		self.processor = AutoProcessor.from_pretrained(self.model_name)
		# SigLIP2 Giant outputs 1536-dim embeddings for image and text
		self.dim = 1536

	@torch.no_grad()
	def image_embed(self, file_or_bytes: Union[bytes, bytearray, io.BytesIO, str]) -> list[float]:
		if isinstance(file_or_bytes, (bytes, bytearray)):
			image = Image.open(io.BytesIO(file_or_bytes)).convert('RGB')
		else:
			image = Image.open(file_or_bytes).convert('RGB')
		inputs = self.processor(images=[image], return_tensors='pt')
		inputs = inputs.to(self.model.device)
		emb = self.model.get_image_features(**inputs)
		return emb.squeeze(0).cpu().tolist()

	@torch.no_grad()
	def image_embed_batch(self, files_or_bytes: List[Union[bytes, bytearray, io.BytesIO, str]]) -> List[List[float]]:
		images = []
		for f in files_or_bytes:
			if isinstance(f, (bytes, bytearray)):
				images.append(Image.open(io.BytesIO(f)).convert('RGB'))
			else:
				images.append(Image.open(f).convert('RGB'))
		inputs = self.processor(images=images, return_tensors='pt')
		inputs = inputs.to(self.model.device)
		emb = self.model.get_image_features(**inputs)
		return emb.cpu().tolist()

	@torch.no_grad()
	def text_embed(self, text: str) -> list[float]:
		inputs = self.processor(text=[text], return_tensors='pt', padding='max_length', max_length=64)
		inputs = inputs.to(self.model.device)
		emb = self.model.get_text_features(**inputs)
		return emb.squeeze(0).cpu().tolist()
