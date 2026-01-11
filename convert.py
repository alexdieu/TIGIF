#!/usr/bin/env python3
"""
CONVERT.PY - Video to TI-83 Premium CE
======================================
FIRST VERSION - UPGRADE OF OLD GIT !
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

def rgb332_convert(frame_bgr):
    """
    Convert BGR frame to RGB332 palette index.
    
    Index format:
      Bits 7-5: Red   (3 bits)
      Bits 4-2: Green (3 bits)
      Bits 1-0: Blue  (2 bits)
    """
    b = frame_bgr[:, :, 0].astype(np.uint16)
    g = frame_bgr[:, :, 1].astype(np.uint16)
    r = frame_bgr[:, :, 2].astype(np.uint16)
    
    r3 = (r >> 5) & 0x07
    g3 = (g >> 5) & 0x07
    b2 = (b >> 6) & 0x03
    
    return ((r3 << 5) | (g3 << 2) | b2).astype(np.uint8)

def extract_frames(video_path, max_frames):
    """Extract video frames and convert to indexed."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return []
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total} frames")
    
    bytes_per_frame = FRAME_WIDTH * FRAME_HEIGHT
    max_possible = MAX_SIZE // bytes_per_frame
    max_frames = min(max_frames or max_possible, max_possible)
    print(f"Extracting up to {max_frames} frames at {FRAME_WIDTH}x{FRAME_HEIGHT}")
    
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        indexed = rgb332_convert(resized)
        frames.append(indexed)
        
        if len(frames) % 10 == 0:
            print(f"  {len(frames)}/{max_frames}...")
    
    cap.release()
    print(f"Extracted {len(frames)} frames")
    return frames

def generate_asm(frames, name):
    """Generate assembly source."""
    n = len(frames)
    
    asm = f"""; Video: {n} frames, {FRAME_WIDTH}x{FRAME_HEIGHT}, scale {SCALE_FACTOR}x

	include 'include/ez80.inc'
	include 'include/tiformat.inc'
	include 'include/ti84pceg.inc'

	format ti executable '{name}'

	call	ti.RunIndicOff
	di

; 1555 palette
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
	ld	hl, frames
	ld	a, {n}
	ld	(cnt), a
.loop:
	push	hl
	call	draw
	pop	hl
	ld	de, {FRAME_WIDTH * FRAME_HEIGHT}
	add	hl, de
	ld	bc, $4000
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
	jr	nz, .loop
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
.row:
	push	af
	ld	b, {SCALE_FACTOR}
.vr:
	push	bc
	push	hl
	ld	c, {FRAME_WIDTH}
.px:
	ld	a, (hl)
	inc	hl
	ld	b, {SCALE_FACTOR}
.hr:
	ld	(de), a
	inc	de
	djnz	.hr
	dec	c
	jr	nz, .px
	pop	hl
	pop	bc
	djnz	.vr
	push	de
	ld	de, {FRAME_WIDTH}
	add	hl, de
	pop	de
	pop	af
	dec	a
	jr	nz, .row
	ret

frames:
"""
    
    for i, f in enumerate(frames):
        asm += f"; F{i}\n"
        data = f.flatten().tolist()
        for j in range(0, len(data), 16):
            asm += "\tdb\t" + ",".join(f"${x:02X}" for x in data[j:j+16]) + "\n"
    
    return asm

def main():
    global FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR
    
    print("=" * 50)
    print("  VIDEO → TI-83 PCE (COLOR)")
    print("=" * 50)
    
    video = input("Video file: ").strip()
    if not os.path.exists(video):
        print("File not found!")
        return 1
    
    name = input("Program name [MYVIDEO]: ").strip().upper()[:8] or "MYVIDEO"
    
    print("\nResolution:")
    print("  1) 40x30  (scale 8x) - ~50 frames")
    print("  2) 64x48  (scale 5x) - ~19 frames")
    print("  3) 80x60  (scale 4x) - ~12 frames")
    r = input("Choice [1]: ").strip()
    
    if r == "2":
        FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR = 64, 48, 5
    elif r == "3":
        FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR = 80, 60, 4
    else:
        FRAME_WIDTH, FRAME_HEIGHT, SCALE_FACTOR = 40, 30, 8
    
    mf = input("Max frames [auto]: ").strip()
    max_frames = int(mf) if mf.isdigit() else None
    
    print()
    frames = extract_frames(video, max_frames)
    if not frames:
        return 1
    
    print("Generating ASM...")
    asm = generate_asm(frames, name)
    
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    
    asm_file = out_dir / f"{name.lower()}.asm"
    with open(asm_file, "w") as f:
        f.write(asm)
    print(f"Saved: {asm_file}")

    fasmg = "./fasmg" if os.name != "nt" else "fasmg.exe"
    out_8xp = out_dir / f"{name.lower()}.8xp"
    
    try:
        result = subprocess.run([fasmg, str(asm_file), str(out_8xp)],
                               capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"\n✓ SUCCESS: {out_8xp}")
            print(f"\nTransfer {name}.8xp to calculator")
            print(f"Run: Asm(prgm{name})")
        else:
            print(f"Compilation error: {result.stderr}")
    except FileNotFoundError:
        print(f"\nfasmg not found. Compile manually:")
        print(f"  fasmg {asm_file} {out_8xp}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
