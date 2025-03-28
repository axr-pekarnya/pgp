from PIL import Image
import struct
import ctypes
import sys

def from_data_to_png(png_path, data_path):
    fin = open(data_path, 'rb')
    (w, h) = struct.unpack('hi', fin.read(8))
    buff = ctypes.create_string_buffer(4 * w * h)
    fin.readinto(buff)
    fin.close()
    img = Image.new('RGBA', (w, h))
    pix = img.load()
    offset = 0
    for j in range(h):
        for i in range(w):
            (r, g, b, a) = struct.unpack_from('cccc', buff, offset)
            pix[i, j] = (ord(r), ord(g), ord(b), ord(a))
            offset += 4
    img.save(png_path)

if __name__ == "__main__":
    frameRate = int(sys.argv[1])

    for i in range(frameRate):
        if i < 10:
            from_data_to_png('data/' + "00" + str(i) + '.png', 'data-bin/' + str(i) + '.data')
        elif i < 100:
            from_data_to_png('data/' + "0" + str(i) + '.png', 'data-bin/' + str(i) + '.data')
        else:
            from_data_to_png('data/' + str(i) + '.png', 'data-bin/' + str(i) + '.data')



