import requests
import subprocess
from PIL import Image
import runpod

from utils import extract_origin_pathname, upload_image, download_url

def run (job, _generator = None):
    # prepare task
    try:
        print('debug', job)

        _input = job.get('input')

        debug = _input.get('debug')
        input_url = _input.get('input_url')
        upload_url = _input.get('upload_url')

        download_url(input_url, './input')

        command = 'python ./ESRGAN/upscale.py ./ESRGAN/4x-UltraSharp.pth'
        subprocess.run(command, shell = True)

        output_image = Image.open('./output/image.png')

        # # output
        output_url = extract_origin_pathname(upload_url)

        output = {
            'type': 'upscale',
            'output_url': output_url
        }

        if debug:
            output_image.save('sample.png')
        else:
            upload_image(upload_url, output_image)

        return output
    # caught http[s] error
    except requests.exceptions.RequestException as e:
        return { 'error': e.args[0] }

runpod.serverless.start({ 'handler': run })
