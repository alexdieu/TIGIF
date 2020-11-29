# TIGIF
A software to convert .mp4 , .mov and .mkv to .8xp 

REQUIRES CLIBS LIBRARY ON YOUR TI(copy clibs.8xg on it with tiConnect) : https://github.com/CE-Programming/libraries/releases/tag/v8.8

WORK ONLY ON : WINDOWS 7/8/10

AND ARTIFICE(if your ti-OS is superior to 5.5): https://yvantt.github.io/arTIfiCE/

THE .8XP IS GENERATED IN `8xp-progs`

USE `convert.py` to generate everything needed and convert it (the conversion is automatic and degrade video to 25x25(in THEORY it could go to 64x64 for vid less than 1 sec) picture , else it would be too big) To CHANGE IT = modify `RESOLUTION_OF_GIF = 25`

*WARNING : `makeVpy.exe` is not usable out of usage*

`makeVpy.exe` was modified by me for the usage of TIGIF(source code and credits at https://github.com/alexdieu/TIGIF#credits-)
### ERRORS

IF U GOT `makeVpy.exe: *** [bin/TIGIF.8xp] Error 1` THIS MEANS THAT YOUR GIF IS TOO LONG OR HAS TOO MUCH FRAMES (25 FPS MAX)

IF U GOT `[success] bin\TIGIF.8xp, 38165 bytes.                                                                                                                                               [ERROR] COMPILATION FAILED !`
          
THIS IS BECAUSE YOU HAVE TO EMPTY 8xp directory , anyways you can get you program at `TIGIF-main\build\bin`

## Credits :

Toolchain for `makeVpy.exe` : https://github.com/CE-Programming/toolchain

Convimg for Image palette quantization : https://github.com/mateoconlechuga/convimg

## RESULTS :

**STARING GIF :**

![start](https://github.com/alexdieu/TIGIF/blob/main/gifDemo.gif)

*OPTIONS :*

`RESOLUTION_OF_GIF = 64`

`SCALE = 4`

**RESULT :**

![start](https://github.com/alexdieu/TIGIF/blob/main/gifDemoR.gif)

*NOTES : It's the recorder that is at 4 frames/sec ! the gif can be up to 25fps on your screen!*

Try mutiples demos on your TI (IN [releases](https://github.com/alexdieu/TIGIF/releases/tag/1) )
