#!/usr/env python3
import http.server
import os
import logging
import io
from PIL import Image
from solve import *
import http.server as server
import json
from corner import *


class HTTPRequestHandler(server.SimpleHTTPRequestHandler):
    """
    SimpleHTTPServer with added bonus of:

    - handle PUT requests
    - log headers in GET request
    """

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write("Success".encode('utf-8'))
        #server.SimpleHTTPRequestHandler.do_GET(self)
        #logging.warning(self.headers)
    def bad_response(self):
        self.send_response(200, 'Created')
        self.end_headers()
        self.wfile.write(json.dumps({}).encode('utf-8'))
    def do_PUT(self):
        try:

            file_length = int(self.headers['Content-Length'])
            this = self.rfile.read(file_length)
            image_data = io.BytesIO(this.split(b'.png"\r\n\r\n')[1].split(b'\r\n--')[0])


            iToImage = getCorners(Image.open(image_data))

            iToLabel = {}

            for i in iToImage:
                r = getPrediction(model, {"input": transform_image_PIL(iToImage[i])})
                iToLabel[i] = r['label']


            print(f"It's {iToLabel}!")
            print(f"Found file size : {file_length}")
            self.send_response(200, 'Created')
            self.end_headers()
            reply_body = json.dumps(iToLabel)
            self.wfile.write(reply_body.encode('utf-8'))
        except:
            self.bad_response()

if __name__ == '__main__':
    server_address = ('localhost', 8002)
    httpd = http.server.HTTPServer(server_address, HTTPRequestHandler)
    httpd.serve_forever()
