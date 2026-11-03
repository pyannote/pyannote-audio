import matplotlib.pyplot as plt
from pyannote.core import Annotation, Segment, notebook

notebook.width = 10
plt.rcParams["figure.figsize"] = (notebook.width, 3)

# only display [0, 20] timerange
notebook.crop = Segment(0, 40)

# plot reference
plt.subplot(211)
reference = Annotation()
reference[Segment(0, 10)] = "A"
reference[Segment(12, 20)] = "B"
reference[Segment(24, 27)] = "A"
reference[Segment(30, 40)] = "C"
notebook.plot_annotation(reference, legend=True, time=False)
plt.gca().text(0.6, 0.15, "reference", fontsize=16)

# plot hypothesis
plt.subplot(212)
hypothesis = Annotation()
hypothesis[Segment(2, 13)] = "a"
hypothesis[Segment(13, 14)] = "d"
hypothesis[Segment(14, 20)] = "b"
hypothesis[Segment(22, 38)] = "c"
hypothesis[Segment(38, 40)] = "d"
notebook.plot_annotation(hypothesis, legend=True, time=True)
plt.gca().text(0.6, 0.15, "hypothesis", fontsize=16)

plt.show()
