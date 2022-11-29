from gatedconvtfworker.gatedconvtfworker import GatedConvTFWorker
from patchmatchworker.patchmatchworker import PatchMatchWorker
from partialconvworker.partialconvworker import PartialConvWorker
from lib.psnr import psnr
from lib.ssim import ssim
from mask import transform_mask

import os
import shutil
from glob import glob
from random import randint
from lib.psnr import psnr
from lib.ssim import ssim
import requests
import json
import base64
import io
from PIL import Image
import time
import config

#static functions
def EndcodeBase64Image(image_path):
    try:
        return base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
    except Exception as e:
        print('encoding failed', e)
        return None

def CreateTempFolder():
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, "Temp")
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    return final_directory

def DownloadImage(url, path):
    try:
        r = requests.get(url, allow_redirects=True)
        open(path, 'wb').write(r.content)
        return True
    except Exception as e:
        print("Failed to download", url, "into", path)
        return False

root = "/home/tuan/inpaint/workers"
temp = root + "/Temp"
outfolder = root + "/output"

def QueryImage(model_identifier = "giberish"):
    try:
        json_data = {}
        r = requests.post(config.WORKER_API_PREFIX + "/queryImage", json=json_data)
        print(config.WORKER_API_PREFIX + "/queryImage", r.status_code, json.dumps(json_data))
        if r.status_code == 200:
            return json.loads(r.text)
        else:
            return None
    except:
        print("Could not have completed queryImage request")
        return None

def GetImageJson(out_local_path, model_identifier, psnr, ssim):
    base64Data = EndcodeBase64Image(out_local_path)
    print(base64Data[:30])
    json_data = {"model_identifier": model_identifier, "outImage": base64Data, "psnr": psnr, "ssim": ssim}
    return json_data
    
def UploadImage(id, images_json):
    url = config.WORKER_API_PREFIX + "/completeImage"
    json_data = {"id": id, "images": images_json}
    r = requests.post(url, json=json_data)
    if r.status_code == 200:
        json_result = json.loads(r.content)
        print(json_result)
        return json_result["code"] == 0
    else:
        print ("upload failed")
        return False

models = [(PartialConvWorker(), False, "partialconv"), (GatedConvTFWorker(), True, "gatedconvtfworker"), (PatchMatchWorker(), True, "patchmatch")]

def Loop():
    while(True):
        #downloading section
        queryResult = QueryImage()
        print(queryResult)
        if queryResult == None:
            print("Result is None")
            time.sleep(1)
        else:
            if queryResult["code"] == 0:
                gt_url = config.IMG_URL_PREFIX + "/" + queryResult["data"]["gt_path"]
                mask_url = config.IMG_URL_PREFIX + "/" + queryResult["data"]["mask_path"]

                gt_local = temp + "/" + "_loop_gt.png"
                raw_mask_local = temp + "/" + "_loop_raw_mask.png"
                mask_local = temp + "/" + "_loop_mask.png"
                out_local = temp + "/_loop_out.png"
                images_to_upload = []
                if DownloadImage(gt_url, gt_local) and DownloadImage(mask_url, raw_mask_local):
                    for model in models:
                        transform_mask(model[1], mask_local, raw_mask_local)
                        if model[0].Compute(gt_local, mask_local, out_local):
                            print (model[2], "successfully computed", gt_url, mask_url)
                            psnr_value = psnr(gt_local, out_local)
                            ssim_value = ssim(gt_local, out_local)
                            images_to_upload.append(GetImageJson(out_local, model[2], psnr_value, ssim_value))
                        else:
                            print (model[2], "failed to compute", gt_url, mask_url)
                    
                    if len(images_to_upload) > 0:
                        #print(json.dumps(images_to_upload))
                        UploadImage(queryResult["data"]["id"], images_to_upload)
            else:
                time.sleep(1)

if __name__ == "__main__":
    Loop()