#!/usr/bin/env python3
"""
CONVERT.PY - Video to TI-83 Premium CE (V4.0 - Maximum Compression)
====================================================================
V4.0 - Phase 2 optimizations for maximum video duration

Features:
- Floyd-Steinberg dithering for better visual quality
- LZ77 compression (better than RLE alone)
- RLE fallback when LZ77 is larger
- Delta encoding with frame skip detection
- Optimized palette matching
- Multiple compression modes (fast/balanced/max)

Target: 5x-10x compression ratio, 20-30+ seconds of video in 60 KB
"""

import cv2
import numpy as np
import os
import sys
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum

# =============================================================================
# CONFIGURATION
# =============================================================================

FRAME_WIDTH = 40
FRAME_HEIGHT = 30
SCALE_FACTOR = 8
MAX_SIZE = 60000          # 60 KB max
CODE_OVERHEAD = 500       # ASM code size estimate
MAX_DATA_SIZE = MAX_SIZE - CODE_OVERHEAD

CPU_FREQ = 48_000_000
CYCLES_PER_ITER = 6

# Compression settings
KEYFRAME_INTERVAL = 15
RLE_MIN_RUN = 3
LZ77_MIN_MATCH = 3
LZ77_MAX_MATCH = 34       # 3-34 bytes (5 bits + 3)
LZ77_WINDOW_SIZE = 255    # 8 bits for offset (look back up to 255 bytes)

class CompressionMode(Enum):
    FAST = "fast"           # RLE only, no dithering
    BALANCED = "balanced"   # RLE + Delta + Dithering
    MAX = "max"             # LZ77 + Delta + Dithering + Frame skip

# =============================================================================
# PALETTE 1555 - TI-83 Premium CE format
# =============================================================================

def compute_palette():
    """Compute exact RGB values for each palette index."""
    palette = []
    for idx in range(256):
        d = idx
        a = idx & 0xC0
        carry = d & 1
        d = d >> 1
        a = ((carry << 7) | (a >> 1)) & 0xFF
        e = a
        a = (31 & idx) | e
        low_byte, high_byte = a, d

        r5 = (high_byte >> 2) & 0x1F
        g_hi = high_byte & 0x03
        g_lo = (low_byte >> 5) & 0x07
        g5 = (g_hi << 3) | g_lo
        b5 = low_byte & 0x1F

        r8 = (r5 << 3) | (r5 >> 2)
        g8 = (g5 << 3) | (g5 >> 2)
        b8 = (b5 << 3) | (b5 >> 2)
        palette.append((r8, g8, b8))

    return np.array(palette, dtype=np.uint8)

print("Building 1555 palette...")
PALETTE = compute_palette()
PALETTE_FLOAT = PALETTE.astype(np.float32)

# Fast LUT for color matching
print("Building LUT...")
LUT_BITS = 5
LUT_SIZE = 1 << LUT_BITS

def find_best(r, g, b):
    diff = PALETTE.astype(np.int32) - np.array([r, g, b], dtype=np.int32)
    dist = np.sum(diff ** 2, axis=1)
    return np.argmin(dist)

LUT = np.zeros((LUT_SIZE, LUT_SIZE, LUT_SIZE), dtype=np.uint8)
for ri in range(LUT_SIZE):
    r8 = (ri << 3) | (ri >> 2)
    for gi in range(LUT_SIZE):
        g8 = (gi << 3) | (gi >> 2)
        for bi in range(LUT_SIZE):
            b8 = (bi << 3) | (bi >> 2)
            LUT[ri, gi, bi] = find_best(r8, g8, b8)
print("  LUT ready!")

# =============================================================================
# DITHERING - Ordered (Bayer) for compression-friendly patterns
# =============================================================================

# Bayer 4x4 dithering matrix (normalized to 0-15, then scaled)
BAYER_4x4 = np.array([
    [ 0,  8,  2, 10],
    [12,  4, 14,  6],
    [ 3, 11,  1,  9],
    [15,  7, 13,  5]
], dtype=np.float32) / 16.0 - 0.5  # Center around 0 (-0.5 to +0.5)

# Scale factor for dithering intensity (lower = less noise = better compression)
DITHER_STRENGTH = 32  # Adjust: 16-48 typical range

def find_closest_color(r, g, b):
    """Find closest palette color to RGB value."""
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    ri = r >> (8 - LUT_BITS)
    gi = g >> (8 - LUT_BITS)
    bi = b >> (8 - LUT_BITS)
    return LUT[ri, gi, bi]

def ordered_dither_bayer(frame_bgr):
    """
    Apply ordered (Bayer) dithering - compression-friendly!

    Unlike Floyd-Steinberg, ordered dithering:
    - Produces deterministic, regular patterns
    - Same pattern applied to all frames (temporal stability)
    - Patterns compress well with RLE/LZ77
    - No error diffusion = no random noise
    """
    h, w = frame_bgr.shape[:2]
    result = np.zeros((h, w), dtype=np.uint8)

    # Tile the Bayer matrix to cover the image
    bayer_h, bayer_w = BAYER_4x4.shape

    for y in range(h):
        for x in range(w):
            # Get Bayer threshold for this position
            threshold = BAYER_4x4[y % bayer_h, x % bayer_w] * DITHER_STRENGTH

            # Get original pixel (BGR)
            b, g, r = frame_bgr[y, x].astype(np.float32)

            # Add dither offset
            r_dith = r + threshold
            g_dith = g + threshold
            b_dith = b + threshold

            # Find closest palette color
            result[y, x] = find_closest_color(r_dith, g_dith, b_dith)

    return result

def ordered_dither_bayer_fast(frame_bgr):
    """
    Vectorized ordered dithering for speed.
    """
    h, w = frame_bgr.shape[:2]

    # Create full-size Bayer threshold matrix
    bayer_h, bayer_w = BAYER_4x4.shape
    tiles_y = (h + bayer_h - 1) // bayer_h
    tiles_x = (w + bayer_w - 1) // bayer_w
    threshold = np.tile(BAYER_4x4, (tiles_y, tiles_x))[:h, :w] * DITHER_STRENGTH

    # Add threshold to all channels
    img = frame_bgr.astype(np.float32)
    img[:, :, 0] += threshold  # B
    img[:, :, 1] += threshold  # G
    img[:, :, 2] += threshold  # R

    # Clip and convert
    img = np.clip(img, 0, 255)

    # Use LUT for fast palette mapping
    r = (img[:, :, 2]).astype(np.uint8) >> (8 - LUT_BITS)
    g = (img[:, :, 1]).astype(np.uint8) >> (8 - LUT_BITS)
    b = (img[:, :, 0]).astype(np.uint8) >> (8 - LUT_BITS)

    return LUT[r, g, b]

# Floyd-Steinberg for comparison (kept but not default)
def floyd_steinberg_dither(frame_bgr):
    """
    Floyd-Steinberg dithering - better visual quality but worse compression.
    Use only for single images or when compression ratio doesn't matter.
    """
    img = frame_bgr.astype(np.float32).copy()
    h, w = img.shape[:2]
    result = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            old_b, old_g, old_r = img[y, x]
            idx = find_closest_color(old_r, old_g, old_b)
            new_r, new_g, new_b = PALETTE[idx]
            result[y, x] = idx

            err_r, err_g, err_b = old_r - new_r, old_g - new_g, old_b - new_b

            if x + 1 < w:
                img[y, x + 1] += [err_b * 7/16, err_g * 7/16, err_r * 7/16]
            if y + 1 < h:
                if x > 0:
                    img[y + 1, x - 1] += [err_b * 3/16, err_g * 3/16, err_r * 3/16]
                img[y + 1, x] += [err_b * 5/16, err_g * 5/16, err_r * 5/16]
                if x + 1 < w:
                    img[y + 1, x + 1] += [err_b * 1/16, err_g * 1/16, err_r * 1/16]

    return result

def convert_frame_simple(frame_bgr):
    """Convert BGR frame to palette indices (no dithering)."""
    r = frame_bgr[:, :, 2] >> (8 - LUT_BITS)
    g = frame_bgr[:, :, 1] >> (8 - LUT_BITS)
    b = frame_bgr[:, :, 0] >> (8 - LUT_BITS)
    return LUT[r, g, b]

def convert_frame(frame_bgr, use_dithering=True):
    """Convert BGR frame to palette indices."""
    if use_dithering:
        # Use ordered dithering (Bayer) - compression-friendly
        return ordered_dither_bayer_fast(frame_bgr)
    return convert_frame_simple(frame_bgr)

# =============================================================================
# COMPRESSION: LZ77 + RLE + DELTA
# =============================================================================

@dataclass
class CompressedFrame:
    """Compressed frame data."""
    data: bytes
    frame_type: str  # "KEY", "DELTA", "SKIP"
    raw_size: int
    method: str = "RLE"  # "RLE" or "LZ77"

    @property
    def ratio(self) -> float:
        return self.raw_size / len(self.data) if self.data else float('inf')

def compress_rle(data: bytes) -> bytes:
    """
    RLE compression - Format:
    - 0x00-0x7F: Literal run, copy next (N+1) bytes (1-128)
    - 0x80-0xFF: Repeat run, repeat next byte (N-0x80+3) times (3-130)
    """
    if not data:
        return b''

    result = bytearray()
    i, n = 0, len(data)

    while i < n:
        # Look for a run
        run_byte = data[i]
        run_len = 1
        while i + run_len < n and data[i + run_len] == run_byte and run_len < 130:
            run_len += 1

        if run_len >= RLE_MIN_RUN:
            result.append(0x80 + (run_len - 3))
            result.append(run_byte)
            i += run_len
        else:
            # Collect literals
            lit_start = i
            while i < n and i - lit_start < 128:
                # Check for upcoming run
                if i + RLE_MIN_RUN <= n:
                    check = data[i]
                    if all(data[i + j] == check for j in range(RLE_MIN_RUN)):
                        break
                i += 1

            lit_len = i - lit_start
            if lit_len > 0:
                result.append(lit_len - 1)
                result.extend(data[lit_start:i])

    return bytes(result)

def compress_lz77(data: bytes) -> bytes:
    """
    LZ77 compression - Format:
    - 0x00-0x7F: Literal run, copy next (N+1) bytes (1-128)
    - 0x80-0xFF: Match, followed by offset byte
      - Length: ((cmd & 0x1F) + 3) = 3-34 bytes
      - Offset: next byte + 1 = 1-256 bytes back

    Designed for fast eZ80 decompression.
    """
    if not data:
        return b''

    result = bytearray()
    i, n = 0, len(data)

    while i < n:
        best_len = 0
        best_offset = 0

        # Search for matches in the sliding window
        search_start = max(0, i - LZ77_WINDOW_SIZE)

        for j in range(search_start, i):
            # Check match length
            match_len = 0
            while (i + match_len < n and
                   match_len < LZ77_MAX_MATCH and
                   data[j + match_len] == data[i + match_len]):
                match_len += 1

            if match_len >= LZ77_MIN_MATCH and match_len > best_len:
                best_len = match_len
                best_offset = i - j

        if best_len >= LZ77_MIN_MATCH:
            # Emit match: 0x80 | (length-3), offset-1
            result.append(0x80 | (best_len - 3))
            result.append(best_offset - 1)
            i += best_len
        else:
            # Collect literals until we find a good match
            lit_start = i
            while i < n and i - lit_start < 128:
                # Quick check for match at current position
                found_match = False
                search_start = max(0, i - LZ77_WINDOW_SIZE)

                for j in range(search_start, i):
                    if i + LZ77_MIN_MATCH <= n:
                        if (data[j:j + LZ77_MIN_MATCH] == data[i:i + LZ77_MIN_MATCH]):
                            found_match = True
                            break

                if found_match:
                    break
                i += 1

            lit_len = i - lit_start
            if lit_len > 0:
                result.append(lit_len - 1)
                result.extend(data[lit_start:i])

    return bytes(result)

def compress_best(data: bytes, use_lz77: bool = True) -> Tuple[bytes, str]:
    """Try both RLE and LZ77, return the smaller result."""
    rle_data = compress_rle(data)

    if not use_lz77:
        return rle_data, "RLE"

    lz77_data = compress_lz77(data)

    if len(lz77_data) < len(rle_data):
        return lz77_data, "LZ77"
    return rle_data, "RLE"

def frames_identical(frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.99) -> bool:
    """Check if two frames are nearly identical."""
    if frame1.shape != frame2.shape:
        return False
    total = frame1.size
    same = np.sum(frame1 == frame2)
    return (same / total) >= threshold

def compress_frame(frame: np.ndarray, prev_frame: Optional[np.ndarray],
                   is_keyframe: bool, use_lz77: bool = True,
                   skip_identical: bool = True) -> CompressedFrame:
    """Compress a single frame with best method."""
    raw_size = frame.size

    # Check for skip frame (identical to previous)
    if skip_identical and prev_frame is not None and frames_identical(frame, prev_frame):
        return CompressedFrame(
            data=b'\x00',  # Single byte marker for skip
            frame_type="SKIP",
            raw_size=raw_size,
            method="SKIP"
        )

    if is_keyframe or prev_frame is None:
        # Keyframe: compress full frame
        raw = frame.flatten().tobytes()
        data, method = compress_best(raw, use_lz77)
        return CompressedFrame(data=data, frame_type="KEY", raw_size=raw_size, method=method)
    else:
        # Delta frame: XOR with previous
        delta = np.bitwise_xor(frame, prev_frame)
        raw = delta.flatten().tobytes()
        data, method = compress_best(raw, use_lz77)

        # Also try keyframe and use if smaller
        key_raw = frame.flatten().tobytes()
        key_data, key_method = compress_best(key_raw, use_lz77)

        if len(key_data) <= len(data):
            return CompressedFrame(data=key_data, frame_type="KEY", raw_size=raw_size, method=key_method)

        return CompressedFrame(data=data, frame_type="DELTA", raw_size=raw_size, method=method)

def compress_video(frames: List[np.ndarray], keyframe_interval: int,
                   mode: CompressionMode = CompressionMode.BALANCED) -> Tuple[List[CompressedFrame], dict]:
    """Compress all frames with specified mode."""
    use_lz77 = (mode == CompressionMode.MAX)
    skip_identical = (mode in [CompressionMode.BALANCED, CompressionMode.MAX])

    compressed = []
    prev_frame = None
    stats = {
        'total_raw': 0,
        'total_compressed': 0,
        'keyframes': 0,
        'delta_frames': 0,
        'skip_frames': 0,
        'rle_frames': 0,
        'lz77_frames': 0
    }

    for i, frame in enumerate(frames):
        is_keyframe = (i % keyframe_interval == 0)
        cf = compress_frame(frame, prev_frame, is_keyframe, use_lz77, skip_identical)
        compressed.append(cf)

        stats['total_raw'] += cf.raw_size
        stats['total_compressed'] += len(cf.data)

        if cf.frame_type == "KEY":
            stats['keyframes'] += 1
        elif cf.frame_type == "DELTA":
            stats['delta_frames'] += 1
        else:
            stats['skip_frames'] += 1

        if cf.method == "LZ77":
            stats['lz77_frames'] += 1
        elif cf.method == "RLE":
            stats['rle_frames'] += 1

        prev_frame = frame

    stats['ratio'] = stats['total_raw'] / stats['total_compressed'] if stats['total_compressed'] > 0 else 0
    return compressed, stats

# =============================================================================
# VIDEO EXTRACTION
# =============================================================================

def calc_delay(fps, fw, fh, scale, has_lz77=False):
    """Calculate delay loop count for target FPS."""
    if fps <= 0:
        return 0x4000

    frame_time = 1.0 / fps
    # LZ77 decompression is slightly slower
    cycles_per_pixel = 40 if has_lz77 else 35
    draw_time = (fw * fh * scale * cycles_per_pixel) / CPU_FREQ
    delay_time = max(0, frame_time - draw_time)

    if delay_time == 0:
        return 0x0100

    iters = int((delay_time * CPU_FREQ) / CYCLES_PER_ITER)
    return max(0x0100, min(0xFFFF, iters))

def extract_frames(path: str, max_frames: Optional[int], target_fps: Optional[float],
                   use_dithering: bool = True) -> Tuple[List[np.ndarray], float]:
    """Extract and convert video frames."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Cannot open video!")
        return [], 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source: {total} frames @ {src_fps:.1f} FPS")

    if target_fps is None:
        target_fps = src_fps

    skip = max(1, int(round(src_fps / target_fps))) if src_fps > target_fps > 0 else 1
    actual_fps = src_fps / skip

    frames = []
    idx = 0
    max_extract = max_frames if max_frames else 1000

    while len(frames) < max_extract:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
            converted = convert_frame(resized, use_dithering)
            frames.append(converted)
            if len(frames) % 20 == 0:
                dither_str = "dithered" if use_dithering else "direct"
                print(f"  Extracted {len(frames)} frames ({dither_str})...", end="\r")
        idx += 1

    cap.release()
    print(f"  Extracted {len(frames)} frames" + " " * 20)
    return frames, actual_fps

def fit_frames_to_size(frames: List[np.ndarray], max_size: int,
                       keyframe_interval: int, mode: CompressionMode) -> Tuple[List[CompressedFrame], dict]:
    """Compress frames and fit as many as possible within size limit."""
    if not frames:
        return [], {}

    compressed, stats = compress_video(frames, keyframe_interval, mode)
    total_size = sum(len(cf.data) for cf in compressed)

    if total_size <= max_size:
        print(f"  All {len(frames)} frames fit! ({total_size:,} bytes)")
        return compressed, stats

    # Binary search for max frames
    low, high = 1, len(frames)
    best_compressed, best_stats = None, None

    while low <= high:
        mid = (low + high) // 2
        test_compressed, test_stats = compress_video(frames[:mid], keyframe_interval, mode)
        test_size = sum(len(cf.data) for cf in test_compressed)

        if test_size <= max_size:
            best_compressed, best_stats = test_compressed, test_stats
            low = mid + 1
        else:
            high = mid - 1

    if best_compressed:
        total = sum(len(cf.data) for cf in best_compressed)
        print(f"  Fit {len(best_compressed)} frames ({total:,} bytes)")

    return best_compressed or [], best_stats or {}

# =============================================================================
# ASM CODE GENERATION
# =============================================================================

def gen_asm(compressed_frames: List[CompressedFrame], name: str, delay: int,
            width: int, height: int, scale: int, use_lz77: bool = False) -> str:
    """Generate optimized ASM code with RLE/LZ77 + Delta decompression."""

    n = len(compressed_frames)
    frame_size = width * height
    screen_width = 320
    line_padding = screen_width - (width * scale)  # Padding needed per line
    clear_size = frame_size * 2  # Total buffer size to clear
    has_skip = any(cf.frame_type == "SKIP" for cf in compressed_frames)

    # Build frame index
    offsets = []
    current_offset = 0
    for cf in compressed_frames:
        offsets.append(current_offset)
        current_offset += len(cf.data)

    # Frame type flags: 0=DELTA, 1=KEY, 2=SKIP
    frame_types = bytes(
        2 if cf.frame_type == "SKIP" else (1 if cf.frame_type == "KEY" else 0)
        for cf in compressed_frames
    )

    # Calculate compression ratio
    total_raw = sum(cf.raw_size for cf in compressed_frames)
    total_compressed = sum(len(cf.data) for cf in compressed_frames)
    ratio = total_raw / total_compressed if total_compressed > 0 else 0

    asm = f"""; {n} frames {width}x{height} scale {scale}x (V4.0 Compressed)
; Compression: {"LZ77" if use_lz77 else "RLE"} + Delta + Skip | Ratio: {ratio:.2f}x

	include 'include/ez80.inc'
	include 'include/tiformat.inc'
	include 'include/ti84pceg.inc'

	format ti executable '{name}'

FRAME_W		equ	{width}
FRAME_H		equ	{height}
FRAME_SIZE	equ	{frame_size}
SCALE		equ	{scale}
NUM_FRAMES	equ	{n}
DELAY		equ	${delay:04X}
SCREEN_W	equ	320
LINE_PAD	equ	{line_padding}
CLEAR_SIZE	equ	{clear_size}

; Buffers in saveSScreen
DECOMP_BUF	equ	0D052C0h
PREV_BUF	equ	DECOMP_BUF + FRAME_SIZE

	call	ti.RunIndicOff
	di

; === Setup palette ===
	ld	hl, ti.mpLcdPalette
	ld	b, 0
.pal:
	ld	d, b
	ld	a, b
	and	a, 192
	srl	d
	rra
	ld	e, a
	ld	a, 31
	and	a, b
	or	a, e
	ld	(hl), a
	inc	hl
	ld	(hl), d
	inc	hl
	inc	b
	jr	nz, .pal

	call	ti.boot.ClearVRAM
	ld	a, ti.lcdBpp8
	ld	(ti.mpLcdCtrl), a

; === Clear BOTH buffers to avoid artifacts ===
	ld	hl, DECOMP_BUF
	ld	de, DECOMP_BUF + 1
	ld	bc, CLEAR_SIZE - 1
	xor	a, a
	ld	(hl), a
	ldir

; === Main loop ===
main:
	ld	hl, frame_offsets
	ld	iy, frame_types
	ld	a, NUM_FRAMES
	ld	(frame_cnt), a

.loop:
	push	hl
	ld	de, (hl)
	ld	hl, data
	add	hl, de

	ld	a, (iy)
	inc	iy
	cp	a, 2
	jr	z, .skip_frame
	or	a, a
	jr	z, .delta_frame

; === Keyframe ===
	ld	de, DECOMP_BUF
	call	decompress
	jr	.copy_to_prev

; === Skip frame (copy from prev) ===
.skip_frame:
	ld	hl, PREV_BUF
	ld	de, DECOMP_BUF
	ld	bc, FRAME_SIZE
	ldir
	jr	.draw

; === Delta frame ===
.delta_frame:
	ld	de, DECOMP_BUF
	call	decompress
	ld	hl, DECOMP_BUF
	ld	de, PREV_BUF
	ld	bc, FRAME_SIZE
.xor_loop:
	ld	a, (de)
	xor	a, (hl)
	ld	(hl), a
	inc	hl
	inc	de
	dec	bc
	ld	a, b
	or	a, c
	jr	nz, .xor_loop

.copy_to_prev:
	ld	hl, DECOMP_BUF
	ld	de, PREV_BUF
	ld	bc, FRAME_SIZE
	ldir

.draw:
	call	draw_frame

	pop	hl
	inc	hl
	inc	hl
	inc	hl

	ld	bc, DELAY
.dly:
	dec	bc
	ld	a, b
	or	a, c
	jr	nz, .dly

	ld	a, (ti.kbdG6)
	bit	ti.kbitClear, a
	jr	nz, quit

	ld	a, (frame_cnt)
	dec	a
	ld	(frame_cnt), a
	jp	nz, .loop
	jp	main

frame_cnt:
	db	0

quit:
	call	ti.ClrScrn
	ld	a, ti.lcdBpp16
	ld	(ti.mpLcdCtrl), a
	call	ti.DrawStatusBar
	ei
	ret

; =============================================================================
; DECOMPRESS - {"LZ77" if use_lz77 else "RLE"} decompression
; Input: HL = source, DE = destination
; =============================================================================
decompress:
	; Calculate end address = DE + FRAME_SIZE
	push	hl
	ld	hl, FRAME_SIZE
	add	hl, de
	ld	(decomp_end), hl
	pop	hl

.loop:
	push	hl
	ld	hl, (decomp_end)
	or	a, a
	sbc	hl, de
	pop	hl
	ret	z
	ret	c

	ld	a, (hl)
	inc	hl
	bit	7, a
	jr	nz, .match

; Literal: copy (A+1) bytes
.literal:
	inc	a
	ld	b, 0
	ld	c, a
	ldir
	jr	.loop

; {"LZ77 Match" if use_lz77 else "RLE Repeat"}
.match:
"""

    if use_lz77:
        # LZ77 decompression: match = copy from earlier in output
        # Fixed for eZ80 24-bit mode
        asm += """\
	and	a, 1Fh
	add	a, 3			; A = length (3-34)
	ld	b, a			; B = length
	ld	a, (hl)			; A = offset-1
	inc	hl
	push	hl			; Save source pointer
	push	de			; Save dest pointer

	; Calculate match source = dest - offset
	inc	a			; A = offset (1-256)
	ld	hl, 0			; Clear all 24 bits of HL
	ld	l, a			; HL = offset (clean 24-bit value)

	pop	de			; DE = dest
	push	de			; Save dest again for writing

	ex	de, hl			; DE = offset, HL = dest
	or	a, a			; Clear carry flag!
	sbc	hl, de			; HL = dest - offset = match source

	pop	de			; DE = dest (for writing)

.copy_match:
	ld	a, (hl)
	ld	(de), a
	inc	hl
	inc	de
	djnz	.copy_match

	pop	hl			; Restore source pointer
	jr	.loop

decomp_end:
	dl	0
"""
    else:
        # RLE decompression: repeat single byte
        asm += """\
	and	a, 7Fh
	add	a, 3			; Count = 3-130
	ld	b, a
	ld	a, (hl)
	inc	hl
.fill:
	ld	(de), a
	inc	de
	djnz	.fill
	jr	.loop

decomp_end:
	dl	0
"""

    # Generate padding code only if needed (when width*scale < 320)
    if line_padding > 0:
        padding_code = """\t; Add padding to reach next screen line
	push	hl
	ld	hl, LINE_PAD
	add	hl, de
	ex	de, hl
	pop	hl
"""
    else:
        padding_code = ""

    asm += f"""
; =============================================================================
; DRAW_FRAME - Scale and display (with screen padding support)
; =============================================================================
draw_frame:
	ld	hl, DECOMP_BUF
	ld	de, ti.vRam
	ld	a, FRAME_H
.row:
	push	af
	ld	b, SCALE
.vscale:
	push	bc
	push	hl
	ld	c, FRAME_W
.pixel:
	ld	a, (hl)
	inc	hl
	ld	b, SCALE
.hscale:
	ld	(de), a
	inc	de
	djnz	.hscale
	dec	c
	jr	nz, .pixel
{padding_code}	pop	hl
	pop	bc
	djnz	.vscale
	push	de
	ld	de, FRAME_W
	add	hl, de
	pop	de
	pop	af
	dec	a
	jr	nz, .row
	ret

; =============================================================================
; DATA
; =============================================================================
frame_offsets:
"""

    for i, offset in enumerate(offsets):
        asm += f"\tdl\t{offset}\n"

    asm += "\nframe_types:\n"
    for i in range(0, len(frame_types), 16):
        chunk = frame_types[i:i+16]
        asm += "\tdb\t" + ",".join(str(b) for b in chunk) + "\n"

    asm += "\ndata:\n"
    for i, cf in enumerate(compressed_frames):
        asm += f"; Frame {i} ({cf.frame_type}, {len(cf.data)}B, {cf.ratio:.1f}x, {cf.method})\n"
        for j in range(0, len(cf.data), 16):
            chunk = cf.data[j:j+16]
            asm += "\tdb\t" + ",".join(f"${b:02X}" for b in chunk) + "\n"

    return asm

# =============================================================================
# MAIN
# =============================================================================

def main():
    global FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR, KEYFRAME_INTERVAL

    print("""
+==============================================================+
|  VIDEO -> TI-83 PCE  V4.0  (Maximum Compression Edition)     |
|  LZ77 + RLE + Delta + Dithering + Frame Skip                 |
+==============================================================+
""")

    video = input("Video path: ").strip()
    if not os.path.exists(video):
        print("File not found!")
        return 1

    name = input("Program name [MYVIDEO]: ").strip().upper()[:8] or "MYVIDEO"

    print("\nResolution:")
    print("  1) 40x30  (scale 8x) - Max frames")
    print("  2) 64x48  (scale 5x) - Balanced")
    print("  3) 80x60  (scale 4x) - High quality")
    print("  4) 106x80 (scale 3x) - Maximum quality")
    r = input("Choice [1]: ").strip()
    if r == "2":
        FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR = 64, 48, 5
    elif r == "3":
        FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR = 80, 60, 4
    elif r == "4":
        FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR = 106, 80, 3
    else:
        FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR = 40, 30, 8

    print("\nCompression mode:")
    print("  1) FAST     - RLE only, no dithering (fastest)")
    print("  2) BALANCED - RLE + Delta + Dithering (recommended)")
    print("  3) MAX      - LZ77 + Delta + Dithering + Skip (best ratio)")
    print("              |-> It's performs better but is highly bug prone and cpu extensive !")
    mode_input = input("Choice [2]: ").strip()
    if mode_input == "1":
        mode = CompressionMode.FAST
    elif mode_input == "3":
        mode = CompressionMode.MAX
    else:
        mode = CompressionMode.BALANCED

    use_dithering = (mode != CompressionMode.FAST)
    use_lz77 = (mode == CompressionMode.MAX)

    fps_input = input("Target FPS [auto]: ").strip()
    target_fps = float(fps_input) if fps_input else None

    max_frames_input = input("Max frames [500]: ").strip()
    max_frames = int(max_frames_input) if max_frames_input.isdigit() else 500

    kf_input = input(f"Keyframe interval [{KEYFRAME_INTERVAL}]: ").strip()
    if kf_input.isdigit():
        KEYFRAME_INTERVAL = int(kf_input)

    print(f"\n--- Extracting frames ({mode.value} mode) ---")
    frames, actual_fps = extract_frames(video, max_frames, target_fps, use_dithering)
    if not frames:
        return 1

    print(f"\n--- Compressing {len(frames)} frames ---")
    compressed, stats = fit_frames_to_size(frames, MAX_DATA_SIZE, KEYFRAME_INTERVAL, mode)

    if not compressed:
        print("Error: Could not compress any frames!")
        return 1

    # Statistics
    print(f"\n{'='*50}")
    print(f"  COMPRESSION STATISTICS ({mode.value.upper()} mode)")
    print(f"{'='*50}")
    print(f"  Total frames:    {len(compressed)}")
    print(f"  - Keyframes:     {stats.get('keyframes', 0)}")
    print(f"  - Delta frames:  {stats.get('delta_frames', 0)}")
    print(f"  - Skip frames:   {stats.get('skip_frames', 0)}")
    if use_lz77:
        print(f"  - LZ77 encoded:  {stats.get('lz77_frames', 0)}")
        print(f"  - RLE encoded:   {stats.get('rle_frames', 0)}")
    print(f"  Raw size:        {stats.get('total_raw', 0):,} bytes")
    print(f"  Compressed:      {stats.get('total_compressed', 0):,} bytes")
    print(f"  Ratio:           {stats.get('ratio', 0):.2f}x")
    print(f"  Duration:        {len(compressed) / actual_fps:.1f} sec @ {actual_fps:.1f} FPS")

    # Capacity comparison
    frame_raw_size = FRAME_WIDTH * FRAME_HEIGHT
    raw_frames = MAX_DATA_SIZE // frame_raw_size
    avg_size = stats.get('total_compressed', 0) / len(compressed) if compressed else frame_raw_size
    compressed_frames = int(MAX_DATA_SIZE / avg_size)

    print(f"\n  --- Capacity ---")
    print(f"  RAW:        {raw_frames} frames ({raw_frames/actual_fps:.1f} sec)")
    print(f"  Compressed: ~{compressed_frames} frames ({compressed_frames/actual_fps:.1f} sec)")
    print(f"  Gain:       {stats.get('ratio', 1):.1f}x more video!")
    print(f"{'='*50}")

    delay = calc_delay(actual_fps, FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR, use_lz77)

    print("\n--- Generating ASM ---")
    asm = gen_asm(compressed, name, delay, FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR, use_lz77)

    out = Path("output")
    out.mkdir(exist_ok=True)
    asm_file = out / f"{name.lower()}.asm"

    with open(asm_file, "w") as f:
        f.write(asm)
    print(f"  Written: {asm_file}")

    # Compile
    fasmg = "./fasmg" if os.name != "nt" else "fasmg.exe"
    out_8xp = out / f"{name.lower()}.8xp"

    print("\n--- Compiling ---")
    try:
        res = subprocess.run([fasmg, str(asm_file), str(out_8xp)],
                            capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            size = out_8xp.stat().st_size
            print(f"  SUCCESS: {out_8xp.name} ({size:,} bytes)")
            print(f"\n  Transfer to calculator: Asm(prgm{name})")
        else:
            print(f"  COMPILE ERROR:\n{res.stderr}")
            return 1
    except FileNotFoundError:
        print(f"  fasmg not found. Compile manually:\n  {fasmg} {asm_file} {out_8xp}")
    except Exception as e:
        print(f"  Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
