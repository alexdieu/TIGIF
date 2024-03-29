import cv2
import os,re, os.path
from PIL import Image
import PIL
import datetime
import shutil
import subprocess
from shutil import Error as er
from time import sleep as wait

count = 0

origi = os.getcwd()

RESOLUTION_OF_GIF = 25 # HIS RESOLUTION (here 25x25)
SCALE = 8 # THE SIZE OF THE GIF OR VIDEO
NAME_OF_PROG = "TIGIF" # NAME OF THE PROGRAM ON YOUR TI
DELAY_BETWEEN_PICTURES = 40 #Delay between pictures in milliseconds . 40 is for 25 FPS .


now = datetime.datetime.now()
debug = []

#REQUIREMENTS
print("VIDEO TO .8XP BY ALEXDIEU FOR TI83 PCE/TI84PCE")
print("VIDEO HAVE TO BE VERY SHORT : LESS THAN 20 SECONDS")
print("This tool requires : ")
print("- BE ON WINDOWS 7/8/10")
print("- Clibs library on your TI")
print("- ARTIFICE if your ti os is superior to 5.0")
input("...")
print("All these conditions are good ? Let's start ! Else go in Readme.md on github")
video = input("video file ?\n")

#GET ALL FRAMES OF THE GIF OR THE VID
def video_to_frames(video, path_output_dir):
    global count
    vidcap = cv2.VideoCapture(video)
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            cv2.imwrite(os.path.join(path_output_dir, 'ti%d.png') % count, image)
            count += 1
            debug.append("[INFO] Succefully writed ti%d.png" % count)
        else:
            print("[WARNING] Failed to write ti%d.png" % count)
            debug.append("[WARNING] Failed to write ti%d.png" % count)
            break
    cv2.destroyAllWindows()
    vidcap.release()

video_to_frames(video, "imgs")

#GETTTING NEW VAR NAMES
g = RESOLUTION_OF_GIF
f = SCALE
v = NAME_OF_PROG

#RESIZING LOW RESOLUTION FOR SPACE
for i in range(0, count):
    try:
        debug.append(f"[INFO] Resizing Image %d.png to ({g}x{g})" % i)
        img = Image.open("imgs//ti%s.png" % i)
        img = img.resize((g, g), PIL.Image.ANTIALIAS)
        img.save("imgs//ti%s.png" % i)
    except:
        print("[WARNING] COULD NOT CONVERT ti%d.png to (25x25)!" % i)
        debug.append("[WARNING] COULD NOT CONVERT ti%d.png to (25x25) !" % i)

#CONVERTING IMAGES TO BINARIES IN C
def convimg(file):
        debug.append("[INFO] WRITTING CONVIMG.YAML")
        try:
            configymaml = open(file,'w')
            configymaml.write("output: c\n")
            configymaml.write("  include-file: gfx.h\n")
            configymaml.write("  palettes:\n")
            configymaml.write("    - global_palette\n")
            configymaml.write("  converts:\n")
            configymaml.write("    - sprites\n")
            configymaml.write("\n")
            configymaml.write("palette: global_palette\n")
            configymaml.write("  fixed-color: {index:0, r:255, g:0, b:128}\n")
            configymaml.write("  fixed-color: {index:1, r:255, g:255, b:255}\n")
            configymaml.write("  images: automatic\n")
            configymaml.write("\n")
            configymaml.write("convert: sprites\n")
            configymaml.write("  palette: global_palette\n")
            configymaml.write("  transparent-color-index: 0\n")
            configymaml.write("  images:\n")
            for i in range(0, count):
                configymaml.write("    - ti%s.png\n" % i)
        except:
            print("[ERROR] COULD NOT WRITE CONVIMG.YAML ! ABORTING OPERATION ... [0!]" % i)
            debug.append("[ERROR] COULD NOT WRITE CONVIMG.YAML ! ABORTING OPERATION ... [0!]" % i)
            exit()
        configymaml.close
        
#CONVERTING IMAGES
convimg("imgs//convimg.yaml")
working_dir = 'imgs'
subprocess.check_call(['convimg.exe'], cwd=working_dir)

files2 = os.listdir('imgs')

#GET ALL FILES .H AND .C
for i in files2:
    destination = "build//src//gfx"
    if '.h' in i or 'c' in i or 'gfx' in i:
        try:
            shutil.move("imgs//%s" %i, destination)
        except:
            mypath = "build//src//gfx"
            print("[WARNING] GFX NOT EMPTY  ! FORMATING ...")
            debug.append("[WARNING] GFX NOT EMPTY  ! FORMATING ...")
            for root, dirs, files in os.walk(mypath):
                for file in files:
                    os.remove(os.path.join(root, file))
            shutil.move("imgs//%s" %i, destination)
    else:
        pass    
#CODE FOR COMPILATION
startcode = '''#include <tice.h>
#include <graphx.h>
#include <stdio.h>
#include <stdlib.h>
#include <keypadc.h>

#include "gfx/gfx.h"

int main(void)
{
	uint8_t key;
	key = kb_ScanGroup(kb_group_6);
	os_ClrHome();
	gfx_Begin();
	gfx_SetPalette(global_palette, sizeof_global_palette, 0);
	gfx_SetTransparentColor(0);
    gfx_FillScreen(1);
	while(kb_ScanGroup(kb_group_1) != kb_2nd) {
'''

k = DELAY_BETWEEN_PICTURES

ende = '''		}
		gfx_End();
		prgm_CleanUp();
    }'''
MAIN = open("build//src//main.c", 'w')
MAIN.write(startcode)
print("[INFO] Writting main.c")
debug.append("[INFO] Writting main.c")
for i in range(0, count):
    #CODE FOR U R PICTURES
    MAIN.write("        gfx_ScaledSprite_NoClip(ti%s, 25, 15, %d, %d);\n" % (i, f, f))
    MAIN.write(f"        delay({k});\n")
MAIN.write(ende)
MAIN.close()

#MAKING THE MAKEFILE
makefile = open("build//makefile", "w")
makef = f'''NAME        ?= {v}
COMPRESSED  ?= NO
ICON        ?= icon.png
DESCRIPTION ?= "TIGIF BY ALEXDIEU FOR TI83PCE/84PCE"'''
makefile.write(makef)
makefile.write("\n\ninclude $(CEDEV)/include/.makefile")
makefile.close()

workdri = 'build'
print("[INFO] BUIDLING ...")
debug.append("[INFO] BUIDLING ...")
#CALLING TO COMPILE THE PROGRAM TO BIN THEN TO 8XP
subprocess.check_call(['make2.exe'], cwd=workdri)

BINARIES = "build//bin"

BINARIESLIST = os.listdir(BINARIES)

debuge = open('DEBUG//DEBUG-LOG(%s_%s_%s).txt' % (now.minute, now.hour, now.day),'w')
       
#GETTING THE 8XP AND MOVING IT
for i in BINARIESLIST:
    destination = "8xp-progs"
    if ".8xp":
        try:
            shutil.move("build//bin//%s" %i, destination)
        except:
            print("[ERROR] COMPILATION FAILED !")
            for element in debug:
                debuge.write(element)
                debuge.write('\n')
            debuge.close()  
            exit()

#WRITTING DEBUG
print("[INFO] Writting debug log")
debug.append("[INFO] Writting debug log")
debug.append("[INFO] DONE")
for element in debug:
     debuge.write(element)
     debuge.write('\n')
debuge.close()  
input("DONE !")
