import matplotlib.pyplot as plt

from pyannote.core import Segment, Timeline, notebook

notebook.width = 10
plt.rcParams["figure.figsize"] = (notebook.width, 2)

# only display [0, 20] timerange
notebook.crop = Segment(0, 20)

# plot timeline
timeline = Timeline()
timeline.add(Segment(1, 5))
timeline.add(Segment(6, 8))
timeline.add(Segment(12, 18))
timeline.add(Segment(7, 20))
notebook.plot_timeline(timeline, time=True)

plt.show()
