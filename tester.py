from gatedconvtfworker.gatedconvtfworker import GatedConvTFWorker
from patchmatchworker.patchmatchworker import PatchMatchWorker
from partialconvworker.partialconvworker import PartialConvWorker
from mask import transform_mask
models = [PartialConvWorker(), GatedConvTFWorker(), PatchMatchWorker()]
hole_types = [False, True, True]

import time

root = "/home/tuan/inpaint/workers"
test_data_path = root + "/" + "testdata"
temp = root + "/Temp"
outfolder = root + "/output"

logpath = root + "/log/verbose.log"
logfile = open(logpath, "w")
def Log(*x):
    print(*x)

def Verbose(*x):
    print(*x, file=logfile)

import os
import shutil
from glob import glob
from random import randint
from lib.psnr import psnr
from lib.ssim import ssim

for subcategory in os.listdir(test_data_path):
    subcategory_time = [0] * len(models)
    sum_psnr = [0] * len(models)
    best_psnr = [-1] * len(models)
    worst_psnr = [100] * len(models)

    sum_ssim = [0] * len(models)
    best_ssim = [-1] * len(models)
    worst_ssim = [100] * len(models)

    min_time = [100000000] * len(models)
    max_time = [-1] * len(models)
    count = 0

    globset = glob(test_data_path + "/" + subcategory + "/*")
    globlength = len(globset)
    for file in globset:
        outpath = temp + "/out.png"#.format(count[i])
        maskpath = temp + "/mask.png"#.format(count[i])
        raw_mask_path = root + "/irregular_mask/{:05d}.png".format(randint(0, 1000000007) % 12000) #irregular
        #raw_mask_path = root + "/regular_mask/mask025.png" #regular
        for i in range(0, len(models)):
            hole_ratio = transform_mask(hole_types[i], maskpath, raw_mask_path)
            st = time.time()
            models[i].Compute(file, maskpath, outpath)
            et = time.time()
            pt = et - st
            subcategory_time[i] += pt
            if pt > max_time[i]:
                max_time[i] = pt
                Log (models[i], subcategory, "Found new max processing time:", max_time[i], file, hole_ratio)
            if pt < min_time[i]:
                min_time[i] = pt
                Log (models[i], subcategory, "Found new min processing time:", min_time[i], file, hole_ratio)

            #shutil.copyfile(file, temp + "/gt{}.png".format(count[i]))
            x = psnr(file, outpath)
            y = ssim(file, outpath)
            sum_psnr[i] += x 
            sum_ssim[i] += y
            if x > best_psnr[i]:
                best_psnr[i] = x 
                Log(models[i], subcategory, "Found new best PSNR:", x, outpath, hole_ratio)
            if x < worst_psnr[i]:
                worst_psnr[i] = x
                Log(models[i], subcategory, "Found new worst PSNR:", x, outpath, hole_ratio)
            
            if y > best_ssim[i]:
                best_ssim[i] = y
                Log(models[i], subcategory, "Found new best SSIM:", y, outpath, hole_ratio)
            if y < worst_ssim[i]:
                worst_ssim[i] = y
                Log(models[i], subcategory, "Found new worst SSIM:", y, outpath, hole_ratio)
            
            Verbose(models[i], subcategory, file, hole_ratio, pt, x, y)

        count += 1
        if count % 20 == 0:
            Log (subcategory, count, "/", globlength, "=", count / globlength, "Time elapsed:", subcategory_time[i], "Time remaining:", subcategory_time[i] / count * (globlength - count))
    
    for i in range(0, len(models)):
        Log (subcategory, models[i], "Average PSNR:", sum_psnr[i], count, sum_psnr[i] / count)
        Log (subcategory, models[i], "Average SSIM:", sum_ssim[i], count, sum_ssim[i] / count)
        Log (subcategory, models[i], "Average processing time", subcategory_time[i], count, subcategory_time[i] / count)
        Log ("")
    Log ("")

            