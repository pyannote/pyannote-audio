import matplotlib.pyplot as plt

from pyannote.core import Segment, notebook

notebook.width = 10
plt.rcParams["figure.figsize"] = (notebook.width, 2)

# only display [0, 20] timerange
notebook.crop = Segment(0, 20)

# plot segment
segment = Segment(5, 15)
notebook.plot_segment(segment, time=True)

plt.show()
