# TIGIF
A software to convert .mp4 , .mov and .mkv to .8xp 

REQUIRES CLIBS LIBRARY ON YOUR TI(copy clibs.8xg on it with tiConnect) : https://github.com/CE-Programming/libraries/releases/tag/v8.8

WORK ONLY ON : WINDOWS 7/8/10

AND ARTIFICE(if your ti-OS is superior to 5.5): https://yvantt.github.io/arTIfiCE/

THE .8XP IS GENERATED IN `8xp-progs`

USE `convert.py` to generate everything needed and convert it (the conversion is automatic and degrade video to 25x25(in THEORY it could go to 64x64 for vid less than 1 sec) picture , else it would be too big)

*WARNING : `makeVpy.exe` is not usable out of usage*

`makeVpy.exe` was modified by me for the usage of TIGIF(source code and credits at https://github.com/alexdieu/TIGIF#credits-)

IF U GOT `makeVpy.exe: *** [bin/TIGIF.8xp] Error 1` THIS MEANS THAT YOUR GIF IS TOO LONG OR HAS TOO MUCH FRAMES (25 FPS MAX)

## Credits :

Toolchain for `makeVpy.exe` : https://github.com/CE-Programming/toolchain

Convimg for Image palette quantization : https://github.com/mateoconlechuga/convimg
