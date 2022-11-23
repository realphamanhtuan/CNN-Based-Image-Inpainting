import requests
import json
import os
import time
import base64
import io
from PIL import Image

class VanisherWorker:
    def Compute(self, gtPath, maskPath, outputPath):
        return True