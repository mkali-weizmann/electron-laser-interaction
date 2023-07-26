from moviepy.editor import VideoFileClip

videoClip = VideoFileClip(r"media/videos/scene/1080p60/partial_movie_files/Microscope/2224377476_2034501960_3403924487.mp4")
# %%
videoClip.write_gif(r"slides\time_modulation.gif", fps=60, program='imageio', opt='OptimizePlus', fuzz=1, loop=0, dispose=True, colors=256, verbose=True, logger=None)

