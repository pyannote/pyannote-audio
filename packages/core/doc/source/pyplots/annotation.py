import matplotlib.pyplot as plt

from pyannote.core import Annotation, Segment, notebook

notebook.width = 10
plt.rcParams["figure.figsize"] = (notebook.width, 2)

# only display [0, 20] timerange
notebook.crop = Segment(0, 20)

# plot annotation
annotation = Annotation()
annotation[Segment(1, 5)] = "Carol"
annotation[Segment(6, 8)] = "Bob"
annotation[Segment(12, 18)] = "Carol"
annotation[Segment(7, 20)] = "Alice"
notebook.plot_annotation(annotation, legend=True, time=True)

plt.show()
