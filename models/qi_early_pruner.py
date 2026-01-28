"""
Query-Image Early Interaction Visual Token Pruner for Qwen3-VL.

This pruner uses Text-to-Image (T2I) similarity for efficient visual token selection,
reducing the number of visual tokens processed by the LLM while preserving relevance
to the query.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class QIEarlyInteractionPruner(nn.Module):
    """
    Query-Image Early Interaction visual token pruner.
    
    Token selection is based on Text-to-Image similarity:
    - Compute cosine similarity between each visual patch and text tokens
    - Take max similarity per patch (how relevant is this patch to the query)
    - Keep top-k patches by T2I score
    
    Args:
        temperature: Temperature for T2I similarity softmax scaling (default: 0.1)
        keep_ratio: Fraction of tokens to keep per image (default: 0.25)
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        keep_ratio: float = 0.25,
    ):
        super().__init__()
        self.temperature = temperature
        self.keep_ratio = keep_ratio
    
    def compute_t2i_similarity(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Text-to-Image similarity scores for each visual token.
        
        For each visual patch, compute the maximum cosine similarity with any
        text token. This identifies patches semantically aligned with the query.
        
        Args:
            text_embeds: Text embeddings, shape (N_text, D)
            image_embeds: Visual patch embeddings, shape (N_patches, D)
            
        Returns:
            T2I similarity scores, shape (N_patches,)
        """
        # Normalize embeddings for cosine similarity
        text_norm = F.normalize(text_embeds, dim=-1)  # (N_text, D)
        image_norm = F.normalize(image_embeds, dim=-1)  # (N_patches, D)
        
        # Compute cosine similarity matrix: (N_text, N_patches)
        similarity = torch.mm(text_norm, image_norm.t())
        
        # Max over text tokens for each patch
        max_sim, _ = similarity.max(dim=0)  # (N_patches,)
        
        return max_sim
    
    @torch.no_grad()
    def forward(
        self,
        text_embeds: torch.Tensor,
        image_embeds_list: List[torch.Tensor],
        keep_ratio: Optional[float] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Prune visual tokens using T2I similarity.
        
        Args:
            text_embeds: Text embeddings for T2I similarity, shape (N_text, D)
            image_embeds_list: List of visual embeddings per image, each (N_i, D)
            keep_ratio: Override default keep_ratio if provided
            
        Returns:
            pruned_embeds_list: List of pruned embeddings per image, each (K_i, D)
            selected_indices_list: List of selected indices per image, each (K_i,)
        """
        if keep_ratio is None:
            keep_ratio = self.keep_ratio
        
        pruned_embeds_list = []
        selected_indices_list = []
        
        for image_embeds in image_embeds_list:
            n_patches = image_embeds.shape[0]
            k = max(1, int(round(keep_ratio * n_patches)))  # At least 1 token
            
            # Compute T2I similarity scores
            t2i_scores = self.compute_t2i_similarity(text_embeds, image_embeds)
            
            # Simply take top-k by T2I score
            _, top_indices = torch.topk(t2i_scores, k, dim=0)
            
            # Sort indices to maintain spatial order
            selected_idx = torch.sort(top_indices)[0]
            
            # Select tokens
            pruned_embeds = image_embeds[selected_idx]
            
            pruned_embeds_list.append(pruned_embeds)
            selected_indices_list.append(selected_idx)
        
        return pruned_embeds_list, selected_indices_list
    
    def forward_with_grad(
        self,
        text_embeds: torch.Tensor,
        image_embeds_list: List[torch.Tensor],
        keep_ratio: Optional[float] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Prune visual tokens with gradient flow through kept tokens.
        
        Same as forward() but allows gradients to flow through selected tokens.
        """
        if keep_ratio is None:
            keep_ratio = self.keep_ratio
        
        pruned_embeds_list = []
        selected_indices_list = []
        
        for image_embeds in image_embeds_list:
            n_patches = image_embeds.shape[0]
            k = max(1, int(round(keep_ratio * n_patches)))
            
            # Compute scores WITHOUT gradients - just for selection
            with torch.no_grad():
                t2i_scores = self.compute_t2i_similarity(text_embeds, image_embeds)
                _, top_indices = torch.topk(t2i_scores, k, dim=0)
                selected_idx = torch.sort(top_indices)[0]
            
            # Select tokens WITH gradients
            pruned_embeds = image_embeds[selected_idx]
            
            pruned_embeds_list.append(pruned_embeds)
            selected_indices_list.append(selected_idx)
        
        return pruned_embeds_list, selected_indices_list
    
    def prune_deepstack_features(
        self,
        deepstack_embeds_list: List[torch.Tensor],
        selected_indices_list: List[torch.Tensor],
        tokens_per_image: List[int],
    ) -> List[torch.Tensor]:
        """
        Apply the same token selection to deepstack features.
        """
        pruned_deepstack_list = []
        
        for deepstack_embeds in deepstack_embeds_list:
            pruned_per_image = []
            offset = 0
            
            for n_i, selected_idx in zip(tokens_per_image, selected_indices_list):
                img_embeds = deepstack_embeds[offset:offset + n_i]
                pruned_img = img_embeds[selected_idx]
                pruned_per_image.append(pruned_img)
                offset += n_i
            
            pruned_deepstack = torch.cat(pruned_per_image, dim=0)
            pruned_deepstack_list.append(pruned_deepstack)
        
        return pruned_deepstack_list


def create_qi_early_pruner(
    temperature: float = 0.1,
    keep_ratio: float = 0.25,
) -> QIEarlyInteractionPruner:
    """
    Factory function to create a QI-Early pruner.
    
    Args:
        temperature: Temperature for T2I similarity (lower = sharper)
        keep_ratio: Fraction of tokens to keep (0.25 = 75% reduction)
        
    Returns:
        Configured QIEarlyInteractionPruner instance
    """
    return QIEarlyInteractionPruner(
        temperature=temperature,
        keep_ratio=keep_ratio,
    )

