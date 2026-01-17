#!/usr/bin/env python3
"""
CONVERT.PY - Video to TI-83 Premium CE
======================================
"""

import cv2
import numpy as np
import os
import sys
import subprocess
from pathlib import Path

# Config
FRAME_WIDTH = 40
FRAME_HEIGHT = 30
SCALE_FACTOR = 8
MAX_SIZE = 60000
CPU_FREQ = 48_000_000
CYCLES_PER_ITER = 6

# =============================================================================
# PALETTE 1555 - Format: xRRRRR GGGGG BBBBB
# =============================================================================

def compute_palette():
    """Compute exact RGB values for each palette index."""
    palette = []
    
    for idx in range(256):
        # Simulate ASM code exactly
        d = idx
        a = idx & 0xC0
        carry = d & 1
        d = d >> 1
        a = ((carry << 7) | (a >> 1)) & 0xFF
        e = a
        a = (31 & idx) | e
        
        low_byte = a
        high_byte = d
        
        # Format 1555: xRRRRR GG | GGG BBBBB
        # high_byte = xRRRRR GG (bits 7-0: x, R4-R0, G4-G3)
        # low_byte  = GGG BBBBB (bits 7-0: G2-G0, B4-B0)
        
        r5 = (high_byte >> 2) & 0x1F  # bits 6-2 of high
        g_hi = high_byte & 0x03       # bits 1-0 of high
        g_lo = (low_byte >> 5) & 0x07 # bits 7-5 of low
        g5 = (g_hi << 3) | g_lo
        b5 = low_byte & 0x1F          # bits 4-0 of low
        
        # Expand 5-bit to 8-bit
        r8 = (r5 << 3) | (r5 >> 2)
        g8 = (g5 << 3) | (g5 >> 2)
        b8 = (b5 << 3) | (b5 >> 2)
        
        palette.append((r8, g8, b8))
    
    return np.array(palette, dtype=np.uint8)

print("Building 1555 palette...")
PALETTE = compute_palette()

# Verify test colors
print(f"  $00 -> RGB{tuple(PALETTE[0x00])} (black)")
print(f"  $E0 -> RGB{tuple(PALETTE[0xE0])} (should be red)")
print(f"  $1C -> RGB{tuple(PALETTE[0x1C])} (should be green)")
print(f"  $03 -> RGB{tuple(PALETTE[0x03])} (should be blue)")
print(f"  $FF -> RGB{tuple(PALETTE[0xFF])} (white)")

# =============================================================================
# Build LUT for fast color matching
# =============================================================================

print("Building LUT (takes ~10 sec)...")

LUT_BITS = 5
LUT_SIZE = 1 << LUT_BITS

def find_best(r, g, b):
    """Find palette index closest to RGB color."""
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
    if ri % 8 == 0:
        pct = ri * 100 // LUT_SIZE
        print(f"  {pct}%", end="\r")

print("  LUT ready!     ")

def convert_frame(frame_bgr):
    """Convert BGR frame to palette indices."""
    r = frame_bgr[:, :, 2] >> (8 - LUT_BITS)
    g = frame_bgr[:, :, 1] >> (8 - LUT_BITS)
    b = frame_bgr[:, :, 0] >> (8 - LUT_BITS)
    return LUT[r, g, b]

# =============================================================================
# Timing & Video
# =============================================================================

def calc_delay(fps, fw, fh, scale):
    if fps <= 0:
        return 0x4000
    frame_time = 1.0 / fps
    draw_time = (fw * fh * scale * 30) / CPU_FREQ
    delay_time = max(0, frame_time - draw_time)
    if delay_time == 0:
        return 0x0100
    iters = int((delay_time * CPU_FREQ) / CYCLES_PER_ITER)
    return max(0x0100, min(0xFFFF, iters))

def extract_frames(path, max_f, target_fps):
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
    target_fps = src_fps / skip
    
    max_possible = MAX_SIZE // (FRAME_WIDTH * FRAME_HEIGHT)
    max_f = min(max_f or max_possible, max_possible)
    
    frames = []
    idx = 0
    while len(frames) < max_f:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
            frames.append(convert_frame(resized))
            if len(frames) % 10 == 0:
                print(f"  {len(frames)}/{max_f}...")
        idx += 1
    
    cap.release()
    print(f"Got {len(frames)} frames")
    return frames, target_fps

# =============================================================================
# ASM
# =============================================================================

def gen_asm(frames, name, delay):
    n = len(frames)
    asm = f"""; {n} frames {FRAME_WIDTH}x{FRAME_HEIGHT} scale {SCALE_FACTOR}x

	include 'include/ez80.inc'
	include 'include/tiformat.inc'
	include 'include/ti84pceg.inc'

	format ti executable '{name}'

	call	ti.RunIndicOff
	di
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
main:
	ld	hl, data
	ld	a, {n}
	ld	(cnt), a
.lp:
	push	hl
	call	draw
	pop	hl
	ld	de, {FRAME_WIDTH * FRAME_HEIGHT}
	add	hl, de
	ld	bc, ${delay:04X}
.dly:
	dec	bc
	ld	a, b
	or	a, c
	jr	nz, .dly
	ld	a, (ti.kbdG6)
	bit	ti.kbitClear, a
	jr	nz, quit
	ld	a, (cnt)
	dec	a
	ld	(cnt), a
	jr	nz, .lp
	jr	main
cnt:
	db	0
quit:
	call	ti.ClrScrn
	ld	a, ti.lcdBpp16
	ld	(ti.mpLcdCtrl), a
	call	ti.DrawStatusBar
	ei
	ret
draw:
	ld	de, ti.vRam
	ld	a, {FRAME_HEIGHT}
.r:
	push	af
	ld	b, {SCALE_FACTOR}
.v:
	push	bc
	push	hl
	ld	c, {FRAME_WIDTH}
.p:
	ld	a, (hl)
	inc	hl
	ld	b, {SCALE_FACTOR}
.h:
	ld	(de), a
	inc	de
	djnz	.h
	dec	c
	jr	nz, .p
	pop	hl
	pop	bc
	djnz	.v
	push	de
	ld	de, {FRAME_WIDTH}
	add	hl, de
	pop	de
	pop	af
	dec	a
	jr	nz, .r
	ret
data:
"""
    for i, f in enumerate(frames):
        asm += f"; F{i}\n"
        d = f.flatten()
        for j in range(0, len(d), 16):
            asm += "\tdb\t" + ",".join(f"${x:02X}" for x in d[j:j+16]) + "\n"
    return asm

# =============================================================================
# Main
# =============================================================================

def main():
    global FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR
    
    print("""
╔════════════════════════════════════════════════╗
║  VIDEO → TI-83 PCE          V2.1               ║
╚════════════════════════════════════════════════╝
""")
    
    video = input("Video: ").strip()
    if not os.path.exists(video):
        print("Not found!")
        return 1
    
    name = input("Name [MYVIDEO]: ").strip().upper()[:8] or "MYVIDEO"
    
    print("\nRes: 1)40x30 2)64x48 3)80x60")
    r = input("[1]: ").strip()
    if r == "2":
        FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR = 64, 48, 5
    elif r == "3":
        FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR = 80, 60, 4
    else:
        FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR = 40, 30, 8
    
    fps = input("FPS [auto]: ").strip()
    fps = float(fps) if fps else None
    
    mf = input("Max frames [auto]: ").strip()
    mf = int(mf) if mf.isdigit() else None
    
    print()
    frames, actual_fps = extract_frames(video, mf, fps)
    if not frames:
        return 1
    
    delay = calc_delay(actual_fps, FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR)
    print(f"Delay: ${delay:04X} for {actual_fps:.1f} FPS")
    
    asm = gen_asm(frames, name, delay)
    
    out = Path("output")
    out.mkdir(exist_ok=True)
    asm_f = out / f"{name.lower()}.asm"
    with open(asm_f, "w") as f:
        f.write(asm)
    
    fasmg = "./fasmg" if os.name != "nt" else "fasmg.exe"
    out_8xp = out / f"{name.lower()}.8xp"
    
    try:
        res = subprocess.run([fasmg, str(asm_f), str(out_8xp)],
                            capture_output=True, text=True, timeout=60)
        if res.returncode == 0:
            print(f"\n✓ {out_8xp.name} - Asm(prgm{name})")
        else:
            print(f"Error: {res.stderr}")
    except:
        print(f"Compile: fasmg {asm_f} {out_8xp}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
