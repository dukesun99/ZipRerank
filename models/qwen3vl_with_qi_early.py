"""
Qwen3-VL with Query-Image Early Interaction Visual Token Pruning.

This module extends Qwen3VLForConditionalGeneration to add training-free
visual token pruning based on query-image similarity for efficient reranking.

CRITICAL: This implementation preserves proper M-RoPE positions for visual tokens.
When visual tokens are pruned, their original spatial positions are retained.
"""

from typing import Optional, Union, List, Tuple, Literal

import torch
import torch.nn as nn

from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
)
from transformers.utils import logging

from .qi_early_pruner import QIEarlyInteractionPruner

logger = logging.get_logger(__name__)


# Type for text extraction mode
TextExtractionMode = Literal["query_only", "all_before_image"]


class Qwen3VLWithQIEarly(Qwen3VLForConditionalGeneration):
    """
    Qwen3VLForConditionalGeneration with Query-Image Early Interaction pruning.
    
    CRITICAL IMPLEMENTATION NOTES:
    - Preserves original M-RoPE positions for pruned visual tokens
    - NO silent fallbacks - errors are raised for any issues
    - Strict validation of all inputs and outputs
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # QI-Early configuration
        self.qi_early_enabled = getattr(config, 'qi_early_enabled', True)
        self.qi_early_keep_ratio = getattr(config, 'qi_early_keep_ratio', 0.25)
        self.qi_early_temperature = getattr(config, 'qi_early_temperature', 0.1)
        self.qi_early_text_mode: TextExtractionMode = getattr(config, 'qi_early_text_mode', 'all_before_image')
        
        # Create pruner (no learnable parameters)
        self.qi_early_pruner = QIEarlyInteractionPruner(
            temperature=self.qi_early_temperature,
            keep_ratio=self.qi_early_keep_ratio,
        )
        
        # Special token IDs for query extraction
        self.query_emb_token_id = getattr(config, 'query_emb_token_id', None)
        
        # Training mode flag - when True, allows gradient flow through kept tokens
        self.qi_early_training_mode = getattr(config, 'qi_early_training_mode', False)
        
        logger.info(f"QI-Early pruning initialized:")
        logger.info(f"  - Enabled: {self.qi_early_enabled}")
        logger.info(f"  - Keep ratio: {self.qi_early_keep_ratio}")
        logger.info(f"  - Temperature: {self.qi_early_temperature}")
        logger.info(f"  - Text mode: {self.qi_early_text_mode}")
        logger.info(f"  - Training mode: {self.qi_early_training_mode}")
    
    def set_qi_early_enabled(self, enabled: bool):
        """Toggle QI-Early pruning at runtime."""
        self.qi_early_enabled = enabled
        logger.info(f"QI-Early pruning {'enabled' if enabled else 'disabled'}")
    
    def set_qi_early_keep_ratio(self, keep_ratio: float):
        """Adjust token keep ratio at runtime."""
        if not 0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.qi_early_keep_ratio = keep_ratio
        self.qi_early_pruner.keep_ratio = keep_ratio
        logger.info(f"QI-Early keep ratio set to {keep_ratio}")
    
    def set_qi_early_text_mode(self, mode: TextExtractionMode):
        """Change text extraction mode at runtime."""
        if mode not in ("query_only", "all_before_image"):
            raise ValueError(f"Invalid text mode: {mode}")
        self.qi_early_text_mode = mode
        logger.info(f"QI-Early text mode set to {mode}")
    
    def set_qi_early_training_mode(self, enabled: bool):
        """
        Toggle QI-Early training mode.
        
        When training mode is enabled:
        - Score computation is done without gradients (selection decision)
        - Token selection allows gradients to flow back to vision encoder/projector
        
        When disabled (default/inference):
        - Entire forward is done without gradients through the pruner
        """
        self.qi_early_training_mode = enabled
        logger.info(f"QI-Early training mode {'enabled' if enabled else 'disabled'}")
    
    def _extract_query_text_embeds(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        mode: Optional[TextExtractionMode] = None,
    ) -> torch.Tensor:
        """
        Extract text embeddings for T2I similarity computation.
        
        STRICT: No fallbacks. Raises error if extraction fails.
        """
        mode = mode or self.qi_early_text_mode
        batch_size = input_ids.shape[0]
        
        if batch_size != 1:
            raise ValueError(f"QI-Early requires batch_size=1, got {batch_size}")
        
        # Get special token IDs
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        vision_end_token_id = self.config.vision_end_token_id
        
        ids = input_ids[0]  # (seq_len,)
        
        # Find first image/video token position
        image_mask = (ids == image_token_id) | (ids == video_token_id)
        image_positions = torch.where(image_mask)[0]
        
        if len(image_positions) == 0:
            raise ValueError("No image tokens found in input_ids. QI-Early requires images.")
        
        first_image_pos = image_positions[0].item()
        
        # Create base exclusion mask (always exclude vision tokens)
        exclude_mask = (
            (ids == image_token_id) |
            (ids == video_token_id) |
            (ids == vision_start_token_id) |
            (ids == vision_end_token_id)
        )
        
        # Also exclude <query_emb> token if present
        if self.query_emb_token_id is not None:
            exclude_mask = exclude_mask | (ids == self.query_emb_token_id)
        
        if mode == "query_only":
            # Find "Query:" marker using heuristic
            query_start = self._find_query_start_position(ids)
            if query_start is None:
                raise ValueError(
                    f"query_only mode: Could not find query marker in input. "
                    f"Use 'all_before_image' mode instead or ensure prompt has 'Query:' marker."
                )
            if query_start >= first_image_pos:
                raise ValueError(
                    f"query_only mode: Query marker found at position {query_start} "
                    f"but first image is at {first_image_pos}. Query must come before images."
                )
            position_mask = (
                (torch.arange(len(ids), device=ids.device) >= query_start) &
                (torch.arange(len(ids), device=ids.device) < first_image_pos)
            )
        else:  # "all_before_image"
            position_mask = torch.arange(len(ids), device=ids.device) < first_image_pos
        
        query_mask = position_mask & ~exclude_mask
        
        # Extract embeddings
        query_embeds = inputs_embeds[0, query_mask]  # (N_query, hidden)
        
        if query_embeds.shape[0] == 0:
            raise ValueError(
                f"No text tokens found for T2I similarity. "
                f"first_image_pos={first_image_pos}, mode={mode}, "
                f"excluded {exclude_mask.sum().item()} tokens"
            )
        
        return query_embeds
    
    def _find_query_start_position(self, input_ids: torch.Tensor) -> Optional[int]:
        """
        Find the position where the user query starts using heuristics.
        
        Looks for newline patterns that typically separate system instruction from query.
        """
        # Get image token position
        image_token_id = self.config.image_token_id
        image_positions = torch.where(input_ids == image_token_id)[0]
        
        if len(image_positions) == 0:
            return None
        
        first_image_pos = image_positions[0].item()
        
        # Common newline tokens in Qwen tokenizers
        newline_tokens = {198, 271, 628, 1432, 720}
        
        # Find the last newline before first image
        last_newline_pos = None
        for pos in range(first_image_pos - 1, -1, -1):
            if input_ids[pos].item() in newline_tokens:
                last_newline_pos = pos + 1  # Start after newline
                break
        
        # Only return if we found a newline that's not at the very beginning
        if last_newline_pos is not None and last_newline_pos > 10:
            return last_newline_pos
        
        return None
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Qwen3VLCausalLMOutputWithPast:
        """
        Forward pass with optional QI-Early visual token pruning.
        
        CRITICAL: When QI-Early is enabled, we:
        1. Compute original M-RoPE positions FIRST using get_rope_index()
        2. Prune visual tokens based on T2I similarity
        3. Build pruned sequence with CORRESPONDING M-RoPE positions preserved
        """
        # If QI-Early disabled or no images, use standard forward
        if not self.qi_early_enabled or pixel_values is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
        
        # === QI-EARLY PRUNING PATH ===
        
        # Validate inputs
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot provide both input_ids and inputs_embeds")
        
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        if batch_size != 1:
            raise ValueError(f"QI-Early requires batch_size=1, got {batch_size}")
        
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required for QI-Early pruning")
        
        # Step 1: Get text embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        # Step 2: Compute ORIGINAL M-RoPE positions using Qwen3VL's method
        original_position_ids, rope_deltas = self.model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask
        )
        
        # Step 3: Get visual embeddings from vision encoder
        image_embeds_list, deepstack_image_embeds = self.model.get_image_features(
            pixel_values, image_grid_thw
        )
        
        # Record original token counts per image
        tokens_per_image = [emb.shape[0] for emb in image_embeds_list]
        total_original_tokens = sum(tokens_per_image)
        
        # Step 4: Extract query text embeddings for T2I similarity
        query_embeds = self._extract_query_text_embeds(inputs_embeds, input_ids)
        
        # Step 5: Apply QI-Early pruning
        if self.qi_early_training_mode:
            pruned_embeds_list, selected_indices_list = self.qi_early_pruner.forward_with_grad(
                text_embeds=query_embeds,
                image_embeds_list=image_embeds_list,
                keep_ratio=self.qi_early_keep_ratio,
            )
        else:
            pruned_embeds_list, selected_indices_list = self.qi_early_pruner(
                text_embeds=query_embeds,
                image_embeds_list=image_embeds_list,
                keep_ratio=self.qi_early_keep_ratio,
            )
        
        # Calculate compression stats
        pruned_tokens_per_image = [emb.shape[0] for emb in pruned_embeds_list]
        total_pruned_tokens = sum(pruned_tokens_per_image)
        
        # Store for ranking loss adjustment
        self.model._last_pruned_per_image = pruned_tokens_per_image
        self.model._last_tokens_per_image = tokens_per_image
        
        logger.debug(
            f"QI-Early: {total_original_tokens} -> {total_pruned_tokens} tokens "
            f"({(1 - total_pruned_tokens/total_original_tokens)*100:.1f}% reduction)"
        )
        
        # Step 6: Prune deepstack features consistently
        if deepstack_image_embeds:
            pruned_deepstack = self.qi_early_pruner.prune_deepstack_features(
                deepstack_image_embeds,
                selected_indices_list,
                tokens_per_image,
            )
        else:
            pruned_deepstack = None
        
        # Step 7: Build pruned sequence with CORRECT M-RoPE positions
        pruned_image_embeds = torch.cat(pruned_embeds_list, dim=0)
        pruned_image_embeds = pruned_image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        
        # Get placeholder mask for original sequence
        original_image_embeds = torch.cat(image_embeds_list, dim=0)
        image_mask, _ = self.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=original_image_embeds
        )
        
        # Build new sequence with pruned tokens AND corresponding positions
        (
            new_inputs_embeds,
            new_attention_mask,
            new_position_ids,
            visual_pos_mask,
        ) = self._build_pruned_sequence_with_positions(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            input_ids=input_ids,
            original_position_ids=original_position_ids,
            image_mask=image_mask,
            pruned_image_embeds=pruned_image_embeds,
            tokens_per_image=tokens_per_image,
            pruned_tokens_per_image=pruned_tokens_per_image,
            selected_indices_list=selected_indices_list,
        )
        
        # Step 8: Forward through language model with pruned inputs
        outputs = self.model.language_model(
            input_ids=None,
            position_ids=new_position_ids,
            attention_mask=new_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=new_inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_mask,
            deepstack_visual_embeds=pruned_deepstack,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        
        # Compute logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            adjusted_labels = self._adjust_labels_for_pruning(
                labels=labels,
                input_ids=input_ids,
                tokens_per_image=tokens_per_image,
                pruned_tokens_per_image=pruned_tokens_per_image,
            )
            loss = self.loss_function(
                logits=logits,
                labels=adjusted_labels,
                vocab_size=self.config.text_config.vocab_size,
            )
        
        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            rope_deltas=rope_deltas,
        )
    
    def _build_pruned_sequence_with_positions(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        original_position_ids: torch.Tensor,
        image_mask: torch.Tensor,
        pruned_image_embeds: torch.Tensor,
        tokens_per_image: List[int],
        pruned_tokens_per_image: List[int],
        selected_indices_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build pruned sequence with CORRECT M-RoPE positions preserved.
        """
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device
        
        # Flatten image_mask to 2D if needed
        if image_mask.dim() == 3:
            image_mask_2d = image_mask[..., 0]
        else:
            image_mask_2d = image_mask
        
        # Get image token ID
        image_token_id = self.config.image_token_id
        
        # Find image token positions in input_ids
        image_positions = (input_ids[0] == image_token_id)
        image_pos_indices = torch.where(image_positions)[0]
        
        if len(image_pos_indices) == 0:
            raise ValueError("No image tokens found in input_ids during sequence building")
        
        # Validate token counts match
        total_image_tokens = len(image_pos_indices)
        expected_tokens = sum(tokens_per_image)
        if total_image_tokens != expected_tokens:
            raise ValueError(
                f"Token count mismatch: found {total_image_tokens} image tokens in input_ids, "
                f"but tokens_per_image sums to {expected_tokens}"
            )
        
        # Group consecutive image positions by image
        image_regions = []
        current_img_start = image_pos_indices[0].item()
        current_img_count = 0
        
        for i, p in enumerate(image_pos_indices.tolist()):
            if i == 0:
                current_img_start = p
                current_img_count = 1
            elif p == image_pos_indices[i-1].item() + 1:
                current_img_count += 1
            else:
                image_regions.append((current_img_start, current_img_start + current_img_count))
                current_img_start = p
                current_img_count = 1
        image_regions.append((current_img_start, current_img_start + current_img_count))
        
        # Validate image regions match expected counts
        if len(image_regions) != len(tokens_per_image):
            raise ValueError(
                f"Found {len(image_regions)} image regions but expected {len(tokens_per_image)}"
            )
        
        for i, ((start, end), expected) in enumerate(zip(image_regions, tokens_per_image)):
            actual = end - start
            if actual != expected:
                raise ValueError(
                    f"Image {i}: region has {actual} tokens but expected {expected}"
                )
        
        # Store image_regions for ranking loss adjustment
        self.model._last_image_ranges = image_regions
        
        # Build new sequence: embeddings, positions, and masks
        new_embeds_parts = []
        new_pos_parts_t = []
        new_pos_parts_h = []
        new_pos_parts_w = []
        visual_mask_parts = []
        
        pos = 0
        
        for img_idx, (img_start, img_end) in enumerate(image_regions):
            # Add text before this image
            if img_start > pos:
                text_embeds = inputs_embeds[0, pos:img_start]
                new_embeds_parts.append(text_embeds)
                
                new_pos_parts_t.append(original_position_ids[0, 0, pos:img_start])
                new_pos_parts_h.append(original_position_ids[1, 0, pos:img_start])
                new_pos_parts_w.append(original_position_ids[2, 0, pos:img_start])
                
                visual_mask_parts.append(torch.zeros(img_start - pos, dtype=torch.bool, device=device))
            
            # Add PRUNED image tokens with their ORIGINAL positions
            n_pruned = pruned_tokens_per_image[img_idx]
            selected_idx = selected_indices_list[img_idx]
            
            # Get pruned embeddings
            pruned_start = sum(pruned_tokens_per_image[:img_idx])
            pruned_img_embeds = pruned_image_embeds[pruned_start:pruned_start + n_pruned]
            new_embeds_parts.append(pruned_img_embeds)
            
            # Get ORIGINAL positions for the selected tokens
            original_img_pos_t = original_position_ids[0, 0, img_start:img_end]
            original_img_pos_h = original_position_ids[1, 0, img_start:img_end]
            original_img_pos_w = original_position_ids[2, 0, img_start:img_end]
            
            new_pos_parts_t.append(original_img_pos_t[selected_idx])
            new_pos_parts_h.append(original_img_pos_h[selected_idx])
            new_pos_parts_w.append(original_img_pos_w[selected_idx])
            
            visual_mask_parts.append(torch.ones(n_pruned, dtype=torch.bool, device=device))
            
            pos = img_end
        
        # Add remaining text after last image
        if pos < seq_len:
            text_embeds = inputs_embeds[0, pos:]
            new_embeds_parts.append(text_embeds)
            
            new_pos_parts_t.append(original_position_ids[0, 0, pos:])
            new_pos_parts_h.append(original_position_ids[1, 0, pos:])
            new_pos_parts_w.append(original_position_ids[2, 0, pos:])
            
            visual_mask_parts.append(torch.zeros(seq_len - pos, dtype=torch.bool, device=device))
        
        # Concatenate all parts
        new_inputs_embeds = torch.cat(new_embeds_parts, dim=0).unsqueeze(0)
        visual_pos_mask = torch.cat(visual_mask_parts, dim=0).unsqueeze(0)
        
        # Build new position_ids tensor: (3, batch, new_seq_len)
        new_pos_t = torch.cat(new_pos_parts_t, dim=0).unsqueeze(0)
        new_pos_h = torch.cat(new_pos_parts_h, dim=0).unsqueeze(0)
        new_pos_w = torch.cat(new_pos_parts_w, dim=0).unsqueeze(0)
        new_position_ids = torch.stack([new_pos_t, new_pos_h, new_pos_w], dim=0)
        
        # Build attention mask
        new_seq_len = new_inputs_embeds.shape[1]
        if attention_mask is not None:
            new_attention_mask = torch.ones(batch_size, new_seq_len, device=device, dtype=attention_mask.dtype)
        else:
            new_attention_mask = torch.ones(batch_size, new_seq_len, device=device)
        
        return new_inputs_embeds, new_attention_mask, new_position_ids, visual_pos_mask
    
    def _adjust_labels_for_pruning(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        tokens_per_image: List[int],
        pruned_tokens_per_image: List[int],
    ) -> torch.Tensor:
        """
        Adjust labels to match the pruned sequence length.
        """
        batch_size, seq_len = labels.shape
        device = labels.device
        
        image_token_id = self.config.image_token_id
        image_positions = (input_ids[0] == image_token_id)
        image_pos_indices = torch.where(image_positions)[0]
        
        if len(image_pos_indices) == 0:
            return labels
        
        # Group by image
        image_regions = []
        current_img_start = image_pos_indices[0].item()
        current_img_count = 0
        
        for i, p in enumerate(image_pos_indices.tolist()):
            if i == 0:
                current_img_start = p
                current_img_count = 1
            elif p == image_pos_indices[i-1].item() + 1:
                current_img_count += 1
            else:
                image_regions.append((current_img_start, current_img_start + current_img_count))
                current_img_start = p
                current_img_count = 1
        image_regions.append((current_img_start, current_img_start + current_img_count))
        
        # Build new labels
        new_label_parts = []
        pos = 0
        
        for img_idx, (img_start, img_end) in enumerate(image_regions):
            # Labels before image
            if img_start > pos:
                new_label_parts.append(labels[0, pos:img_start])
            
            # Pruned image region: all -100 (ignore)
            n_pruned = pruned_tokens_per_image[img_idx]
            new_label_parts.append(torch.full((n_pruned,), -100, dtype=labels.dtype, device=device))
            
            pos = img_end
        
        # Labels after last image
        if pos < seq_len:
            new_label_parts.append(labels[0, pos:])
        
        new_labels = torch.cat(new_label_parts, dim=0).unsqueeze(0)
        return new_labels

