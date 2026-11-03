import matplotlib.pyplot as plt

from pyannote.core import Annotation, Segment, Timeline, notebook

notebook.width = 10
plt.rcParams["figure.figsize"] = (notebook.width, 3)

# only display [0, 20] timerange
notebook.crop = Segment(0, 20)

# plot annotation
plt.subplot(211)
annotation = Annotation()
annotation[Segment(1, 5)] = "Carol"
annotation[Segment(6, 8)] = "Bob"
annotation[Segment(12, 18)] = "Carol"
annotation[Segment(7, 20)] = "Alice"
notebook.plot_annotation(annotation, legend=True, time=False)

# plot timeline
plt.subplot(212)
timeline = Timeline()
timeline.add(Segment(1, 5))
timeline.add(Segment(6, 8))
timeline.add(Segment(12, 18))
timeline.add(Segment(7, 20))
notebook.plot_timeline(timeline, time=True)

plt.show()
