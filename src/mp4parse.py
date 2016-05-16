import subprocess as sp
import PIL.Image as Image

from utils import *

FFMPEG_BIN = "ffmpeg"


def get_pipe(path):
    command = [FFMPEG_BIN,
            '-i', path,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    return pipe


def next_frame(pipe, frame_size):
    length = frame_size[0] * frame_size[1] * frame_size[2]
    raw_image = pipe.stdout.read(length)
    if not raw_image:
        return None
    image = np.fromstring(raw_image, dtype='uint8')
    image = image.reshape(frame_size)
    pipe.stdout.flush()
    return image


def main():
    """
    Args: video_file output_dir in_width in_height out_width out_height frame_freq
    """
    args = sys.argv
    video_path = args[1]
    output_dir = args[2]
    in_width = int(args[3])
    in_height = int(args[4])
    out_width = int(args[5])
    out_height = int(args[6])
    frame_freq = int(args[7])

    frame_size = (in_height, in_width, 3)  # reversed
    output_size = (out_width, out_height)

    create_dir(output_dir)

    with open(os.path.join(output_dir, 'info.txt'), 'w') as info_f:
        info_f.write('Frame size: %i x %i\nFrame frequency: %i\n' % (out_width, out_height, frame_freq))

    pipe = get_pipe(video_path)
    i = 0
    while True:
        # if i % 500 == 0:
            # my_print('\rCurrent frame: %i' % i)
        image = next_frame(pipe, frame_size)
        if image is None:
            break
        if i % frame_freq == 0:
            resized = Image.fromarray(image).resize(output_size, Image.ANTIALIAS)
            image_path = os.path.join(output_dir, str(i / frame_freq) + '.png')
            resized.save(image_path, 'PNG')
        i += 1

    # my_print('\rDone\n')

main()

