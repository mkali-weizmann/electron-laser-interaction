from moviepy.editor import VideoFileClip
import imageio
import sys
input_path = r"media/videos/scene/1080p60/partial_movie_files/Microscope/2224377476_3238693359_3403924487.mp4"
outputpath = r"slides\time_modulation_new.gif"

# videoClip = VideoFileClip(r"media/videos/scene/1080p60/partial_movie_files/Microscope/2224377476_2911817037_3403924487.mp4")
# %%
# videoClip.write_gif(r"slides\time_modulation.gif", fps=60, program='imageio')

# %%


class TargetFormat(object):
    GIF = ".gif"
    MP4 = ".mp4"
    AVI = ".avi"

def convertFile(inputpath, targetFormat):
    """Reference: http://imageio.readthedocs.io/en/latest/examples.html#convert-a-movie"""

    print("converting\r\n\t{0}\r\nto\r\n\t{1}".format(inputpath, outputpath))
    reader = imageio.get_reader(inputpath)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(outputpath, duration=1000*1/50)
    for i,im in enumerate(reader):
        sys.stdout.write("\rframe {0}".format(i))
        sys.stdout.flush()
        writer.append_data(im)
    print("\r\nFinalizing...")
    writer.close()
    print("Done.")

convertFile(input_path, TargetFormat.GIF)
