from PIL import Image
import struct
import sys

def convert_jpg_to_custom_format(input_path, output_path):
    image = Image.open(input_path).convert("RGBA")
    width, height = image.size
    pixels = list(image.getdata())
    
    with open(output_path, "wb") as f:
        f.write(struct.pack("<II", width, height))
        
        for pixel in pixels:
            r, g, b, a = pixel
            f.write(struct.pack("<BBBB", r, g, b, a))
    
def convert_custom_to_jpg(input_path, output_path):
    with open(input_path, "rb") as f:
        width, height = struct.unpack("<II", f.read(8))
        
        pixels = []
        for _ in range(width * height):
            r, g, b, a = struct.unpack("<BBBB", f.read(4))
            pixels.append((r, g, b))
        
        image = Image.new("RGB", (width, height))
        image.putdata(pixels)
        image.save(output_path, "JPEG")
    
#input_file = "/mnt/data/изображение.jpg"
#output_file = "/mnt/data/изображение.bin"
#convert_jpg_to_custom_format(input_file, output_file)

input_bin = "output2.data"
output_jpg = "data-2.jpg"
convert_custom_to_jpg(input_bin, output_jpg)
