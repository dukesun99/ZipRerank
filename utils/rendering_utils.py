"""
Document Renderer: Converts text documents to images for optical encoding.

Supports multiple rendering strategies:
1. Fixed canvas size with text wrapping
2. Dynamic sizing based on content length
3. Multi-page rendering for long documents
"""

import os
import hashlib
from typing import Optional, Tuple, List
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np


class DocumentRenderer:
    """Renders text documents as images for vision encoders."""
    
    def __init__(
        self,
        image_size: int = 640,
        font_size: int = 12,
        font_path: Optional[str] = None,
        padding: int = 20,
        line_spacing: int = 4,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        cache_dir: Optional[str] = None,
        fill_canvas: bool = False,
        fill_threshold: float = 0.7,
        sequential_patches: bool = False,
        patch_size: int = 16,
        downsample_ratio: int = 4,
        sequential_patch_padding: int = 0,
    ):
        """
        Initialize document renderer.
        
        Args:
            image_size: Target image size for rendering (e.g. 280, 640, 1024)
            font_size: Font size for text (initial size if fill_canvas=True)
            font_path: Path to TTF font file (uses default if None)
            padding: Padding around text in pixels
            line_spacing: Spacing between lines
            bg_color: Background RGB color
            text_color: Text RGB color
            normalize_mean: Mean for normalization (DeepSeek-OCR uses 0.5, 0.5, 0.5)
            normalize_std: Std for normalization (DeepSeek-OCR uses 0.5, 0.5, 0.5)
            cache_dir: Directory to cache rendered images
            fill_canvas: If True, dynamically adjust font size to fill the entire canvas
            fill_threshold: Canvas fill ratio (0.0-1.0) to exit early (default: 0.7 = 70%)
            sequential_patches: If True, render text sequences patch-by-patch for complete sequences per patch
            patch_size: Vision encoder patch size (default 16 for DeepSeek-OCR)
            downsample_ratio: Vision encoder downsample ratio (default 4)
            sequential_patch_padding: Padding (in pixels) inside each patch when sequential_patches is True (default 0)
        """
        self.image_size = image_size
        self.font_size = font_size
        self.padding = padding
        self.line_spacing = line_spacing
        self.bg_color = bg_color
        self.text_color = text_color
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.cache_dir = cache_dir
        self.fill_canvas = fill_canvas
        self.fill_threshold = fill_threshold
        self.sequential_patches = sequential_patches
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.font_path = font_path
        self.sequential_patch_padding = max(0, sequential_patch_padding)
        
        # Calculate patch grid dimensions
        # For 640x640 with patch_size=16, downsample_ratio=4: we get 10x10 grid
        import math
        self.patch_grid_size = math.ceil((image_size // patch_size) / downsample_ratio)
        
        # Load font
        self.font = self._load_font(font_size)
        
        # Create cache directory if specified
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _load_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        """Load font with specified size."""
        try:
            if self.font_path and os.path.exists(self.font_path):
                return ImageFont.truetype(self.font_path, font_size)
            else:
                # Try to load a monospace font
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        except:
            # Fall back to default font
            return ImageFont.load_default()
    
    def _compute_text_hash(self, text: str) -> str:
        """Compute hash of text for caching."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, text: str, size: int) -> Optional[Path]:
        """Get cache file path for a rendered document."""
        if not self.cache_dir:
            return None
        text_hash = self._compute_text_hash(text)
        filename = f"{text_hash}_{size}.png"
        return Path(self.cache_dir) / filename
    
    def _wrap_text(self, text: str, max_width: int, font: Optional[ImageFont.FreeTypeFont] = None) -> List[str]:
        """Wrap text to fit within max_width pixels with optional hyphenation."""
        if font is None:
            font = self.font

        if max_width <= 0:
            return [text] if text else []
        
        tokens: List[Tuple[str, bool]] = []  # (token, glue_to_previous)

        for word in text.split():
            bbox = font.getbbox(word)
            width = bbox[2] - bbox[0]

            if width <= max_width:
                tokens.append((word, False))
            else:
                segments = self._hyphenate_word(word, max_width, font)
                if not segments:
                    continue
                for idx, segment in enumerate(segments):
                    glue = idx > 0  # subsequent segments glue to previous (no space)
                    tokens.append((segment, glue))

        lines: List[str] = []
        current_line = ''

        for token, glue in tokens:
            while True:
                if not current_line:
                    candidate = token
                else:
                    if glue:
                        candidate = current_line + token
                    else:
                        candidate = current_line + ' ' + token

                bbox = font.getbbox(candidate)
                width = bbox[2] - bbox[0]

                if width <= max_width or not current_line:
                    current_line = candidate
                    break
                else:
                    # Current line can't accommodate the token; flush and retry
                    lines.append(current_line)
                    current_line = ''
                    glue = False

        if current_line:
            lines.append(current_line)

        return lines
    
    def _hyphenate_word(
        self,
        word: str,
        max_width: int,
        font: ImageFont.FreeTypeFont,
    ) -> List[str]:
        """
        Break a single word into multiple segments that each fit within max_width.
        Hyphenate between segments to avoid overflow.
        """
        if not word:
            return []

        segments: List[str] = []
        remaining = word

        # Pre-compute hyphen width
        hyphen_width = font.getbbox('-')
        hyphen_width = (hyphen_width[2] - hyphen_width[0]) if hyphen_width else 0

        while remaining:
            best_segment = ''
            best_len = 0
            n = len(remaining)

            for i in range(1, n + 1):
                segment = remaining[:i]
                is_last = (i == n)
                candidate = segment if is_last else segment + '-'
                bbox = font.getbbox(candidate)
                width = bbox[2] - bbox[0]

                if width <= max_width:
                    best_segment = candidate
                    best_len = i
                else:
                    break

            if best_len == 0:
                # Force at least one character with hyphen if possible
                segment = remaining[0]
                if len(remaining) > 1 and max_width > hyphen_width:
                    best_segment = segment + '-'
                else:
                    best_segment = segment
                best_len = 1

            segments.append(best_segment)
            remaining = remaining[best_len:]

        # Remove trailing hyphen from last segment, if present
        if segments and segments[-1].endswith('-'):
            segments[-1] = segments[-1][:-1]

        return segments
    
    def _calculate_optimal_font_size(
        self,
        text: str,
        target_size: int,
        min_font_size: int = 6,
        max_font_size: int = 48,
    ) -> Tuple[int, ImageFont.FreeTypeFont, List[str]]:
        """
        Binary search to find optimal font size that fills the canvas.
        
        Args:
            text: Text to render
            target_size: Target image size
            min_font_size: Minimum font size to try
            max_font_size: Maximum font size to try
            
        Returns:
            (optimal_font_size, font, wrapped_lines)
        """
        max_text_width = target_size - 2 * self.padding
        max_height = target_size - 2 * self.padding
        
        best_font_size = self.font_size
        best_font = self.font
        best_lines = self._wrap_text(text, max_text_width)
        
        # Early exit: if default font already fills >= threshold, skip binary search
        bbox = best_font.getbbox('Ay')
        line_height = bbox[3] - bbox[1] + self.line_spacing
        total_height = len(best_lines) * line_height
        fill_ratio = total_height / max_height
        
        if fill_ratio >= self.fill_threshold:
            # Text already fills enough of the canvas, no need to search
            return best_font_size, best_font, best_lines
        
        # Binary search for optimal font size
        low, high = min_font_size, max_font_size
        
        while low <= high:
            mid = (low + high) // 2
            font = self._load_font(mid)
            
            # Wrap text with this font size
            lines = self._wrap_text(text, max_text_width, font)
            
            # Calculate total height
            bbox = font.getbbox('Ay')
            line_height = bbox[3] - bbox[1] + self.line_spacing
            total_height = len(lines) * line_height
            
            if total_height <= max_height:
                # This font size fits, try larger
                best_font_size = mid
                best_font = font
                best_lines = lines
                low = mid + 1
            else:
                # This font size is too large, try smaller
                high = mid - 1
        
        return best_font_size, best_font, best_lines
    
    def _split_text_into_sequences(self, text: str, separator: str = None) -> List[str]:
        """
        Split text into sequences for sequential patch rendering.
        
        By default, splits on double newlines or treats each sentence as a sequence.
        
        Args:
            text: Input text
            separator: Optional custom separator
            
        Returns:
            List of text sequences
        """
        if separator:
            sequences = text.split(separator)
        else:
            # Try to split intelligently: by paragraphs first, then by sentences
            # Split by double newlines (paragraphs)
            paragraphs = text.split('\n\n')
            sequences = []
            for para in paragraphs:
                # If paragraph is very short, keep it as is
                if len(para) < 100:
                    sequences.append(para.strip())
                else:
                    # Split long paragraphs into sentences
                    import re
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    sequences.extend(s.strip() for s in sentences if s.strip())
        
        return [s for s in sequences if s]  # Filter empty strings
    
    def _render_sequential_patches(
        self,
        text: str,
        size: int,
    ) -> Image.Image:
        """
        Render text in sequential patch mode.
        
        In this mode, text sequences are rendered patch-by-patch, ensuring
        complete sequences fit within patch boundaries rather than being split.
        Patches are filled left-to-right, top-to-bottom.
        
        Note: Uses configurable minimal padding (default 0px) to maximize space per patch.
        
        Args:
            text: Text to render
            size: Target image size
            
        Returns:
            Rendered image
        """
        # Create blank image
        img = Image.new('RGB', (size, size), color=self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Calculate patch dimensions in pixels
        # Each patch is a square region in the image
        patch_pixel_size = size // self.patch_grid_size
        
        # Use configurable padding for sequential patch mode to maximize space
        patch_padding = max(0, self.sequential_patch_padding)
        
        # Split text into sequences
        sequences = self._split_text_into_sequences(text)
        
        # Calculate line height
        bbox = self.font.getbbox('Ay')
        line_height = bbox[3] - bbox[1] + self.line_spacing
        
        # Track current patch position (filling left-to-right, top-to-bottom)
        current_patch_idx = 0  # Linear patch index
        max_patches = self.patch_grid_size * self.patch_grid_size
        
        # Track position within current patch
        patch_y_offset = patch_padding  # Y offset within the current patch
        
        seq_idx = 0
        while seq_idx < len(sequences) and current_patch_idx < max_patches:
            sequence = sequences[seq_idx]
            
            # Calculate current patch coordinates
            patch_row = current_patch_idx // self.patch_grid_size
            patch_col = current_patch_idx % self.patch_grid_size
            
            # Patch boundaries
            patch_x_start = patch_col * patch_pixel_size
            patch_y_start = patch_row * patch_pixel_size
            patch_x_end = patch_x_start + patch_pixel_size
            patch_y_end = patch_y_start + patch_pixel_size
            
            # Text rendering area within patch (with minimal padding)
            text_x = patch_x_start + patch_padding
            text_width = patch_pixel_size - 2 * patch_padding
            text_y_start = patch_y_start + patch_padding
            text_y_end = patch_y_end - patch_padding
            
            # Wrap this sequence to fit patch width
            lines = self._wrap_text(sequence, text_width)
            
            # Calculate height needed for this sequence
            seq_height = len(lines) * line_height
            
            # Calculate remaining height in current patch
            current_y = patch_y_start + patch_y_offset
            remaining_height = text_y_end - current_y
            
            # Check if sequence fits in current patch
            if seq_height <= remaining_height:
                # Render the sequence in current patch
                for line in lines:
                    if current_y + line_height > text_y_end:
                        break
                    draw.text((text_x, current_y), line, fill=self.text_color, font=self.font)
                    current_y += line_height
                
                # Update position for next sequence
                patch_y_offset = current_y - patch_y_start + line_height // 2  # Add spacing
                seq_idx += 1
                
            else:
                # Sequence doesn't fit in current patch
                lines_that_fit = 0

                if remaining_height >= line_height:
                    # Render as many lines as possible in the current patch
                    lines_that_fit = min(len(lines), int(remaining_height // line_height))

                    for line in lines[:lines_that_fit]:
                        draw.text((text_x, current_y), line, fill=self.text_color, font=self.font)
                        current_y += line_height

                # Move to next patch and reset offset
                current_patch_idx += 1
                patch_y_offset = patch_padding

                if lines_that_fit == 0:
                    # Couldn't render anything; try the same sequence in the next patch
                    continue

                if lines_that_fit < len(lines):
                    # There are remaining lines; re-queue them as the current sequence
                    remaining_lines = lines[lines_that_fit:]
                    sequences[seq_idx] = ' '.join(remaining_lines)
                else:
                    # Entire sequence rendered across patches
                    seq_idx += 1
 
        return img
    
    def render(
        self,
        text: str,
        size: Optional[int] = None,
        use_cache: bool = False,  # Disabled to save disk space
    ) -> Image.Image:
        """
        Render text as an image.
        
        Args:
            text: Text content to render
            size: Target size (uses self.image_size if None)
            use_cache: Whether to use cached images
            
        Returns:
            PIL Image of rendered document
        """
        if size is None:
            size = self.image_size
        
        # Check cache
        if use_cache:
            cache_path = self._get_cache_path(text, size)
            if cache_path and cache_path.exists():
                try:
                    # Verify image can be opened before returning
                    img = Image.open(cache_path)
                    img.verify()  # Verify it's a valid image
                    # Re-open after verify (verify closes the file)
                    img = Image.open(cache_path).convert('RGB')
                    return img
                except Exception as e:
                    # Cache file is corrupted, delete and re-render
                    try:
                        cache_path.unlink()
                    except:
                        pass
                    # Continue to render below
        
        # Use sequential patch rendering if enabled
        if self.sequential_patches:
            img = self._render_sequential_patches(text, size)
        # Use dynamic font sizing if enabled
        elif self.fill_canvas:
            img = self._render_with_fill_canvas(text, size)
        # Standard rendering
        else:
            img = self._render_standard(text, size)
        
        # Save to cache if enabled (use atomic write to avoid race conditions)
        if use_cache and cache_path:
            # Write to temporary file first, then atomically rename
            # Use .png.tmp to ensure PIL recognizes the format
            temp_path = Path(str(cache_path) + '.tmp')
            try:
                # PIL needs format specified for non-standard extensions
                img.save(temp_path, format='PNG')
                # Atomic rename (will overwrite if another process finished first)
                temp_path.replace(cache_path)
            except Exception as e:
                # Clean up temp file if save failed
                if temp_path.exists():
                    temp_path.unlink()
                # Don't raise - just continue without caching
                pass
        
        return img
    
    def _render_standard(
        self,
        text: str,
        size: int,
    ) -> Image.Image:
        """Standard rendering with fixed font size."""
        # Create blank image
        img = Image.new('RGB', (size, size), color=self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Wrap text to fit canvas
        max_text_width = size - 2 * self.padding
        lines = self._wrap_text(text, max_text_width)
        
        # Calculate line height
        bbox = self.font.getbbox('Ay')  # Use characters with ascenders/descenders
        line_height = bbox[3] - bbox[1] + self.line_spacing
        
        # Draw text line by line
        y = self.padding
        max_lines = (size - 2 * self.padding) // line_height
        
        for i, line in enumerate(lines):
            if i >= max_lines:
                # Add ellipsis if text is truncated
                draw.text((self.padding, y), "...", fill=self.text_color, font=self.font)
                break
            draw.text((self.padding, y), line, fill=self.text_color, font=self.font)
            y += line_height
        
        return img
    
    def _render_with_fill_canvas(
        self,
        text: str,
        size: int,
    ) -> Image.Image:
        """
        Render with dynamic font sizing to fill the entire canvas.
        
        This mode automatically adjusts the font size to maximize space utilization,
        ensuring minimal empty space at the bottom of the canvas.
        """
        # Find optimal font size
        optimal_font_size, optimal_font, lines = self._calculate_optimal_font_size(text, size)
        
        # Create blank image
        img = Image.new('RGB', (size, size), color=self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Calculate line height with optimal font
        bbox = optimal_font.getbbox('Ay')
        line_height = bbox[3] - bbox[1] + self.line_spacing
        
        # Draw text line by line
        y = self.padding
        max_height = size - self.padding
        
        for i, line in enumerate(lines):
            if y + line_height > max_height:
                # Add ellipsis if we somehow overflow (shouldn't happen with proper calculation)
                if y + line_height // 2 <= max_height:
                    draw.text((self.padding, y), "...", fill=self.text_color, font=optimal_font)
                break
            draw.text((self.padding, y), line, fill=self.text_color, font=optimal_font)
            y += line_height
        
        return img
    
    def render_to_tensor(
        self,
        text: str,
        size: Optional[int] = None,
        normalize: bool = True,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        """
        Render text and convert to normalized tensor.
        
        Args:
            text: Text content to render
            size: Target size
            normalize: Whether to normalize with mean/std
            dtype: Output tensor dtype (default: float16 for GPU efficiency)
            
        Returns:
            Tensor of shape [3, size, size] normalized for vision encoder
        """
        img = self.render(text, size=size)
        
        # Convert to tensor [3, H, W]
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Normalize
        if normalize:
            for c in range(3):
                img_tensor[c] = (img_tensor[c] - self.normalize_mean[c]) / self.normalize_std[c]
        
        # Convert to target dtype
        return img_tensor.to(dtype=dtype)
    
    def render_batch(
        self,
        texts: List[str],
        size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Render batch of texts to tensor.
        
        Args:
            texts: List of text documents
            size: Target size
            
        Returns:
            Tensor of shape [batch, 3, size, size]
        """
        tensors = [self.render_to_tensor(text, size=size) for text in texts]
        return torch.stack(tensors, dim=0)

