# The MIT License (MIT)

# Copyright (c) 2014-2019 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

"""
#############
Visualization
#############

:class:`pyannote.core.Segment`, :class:`pyannote.core.Timeline`,
:class:`pyannote.core.Annotation` and :class:`pyannote.core.SlidingWindowFeature`
instances can be directly visualized in notebooks.

You will however need to install ``pyannote.core``'s additional dependencies
for notebook representations (namely, matplotlib):


.. code-block:: bash

    pip install pyannote.core[notebook]


Segments
--------

.. code-block:: ipython

  In [1]: from pyannote.core import Segment

  In [2]: segment = Segment(start=5, end=15)
    ....: segment

.. plot:: pyplots/segment.py


Timelines
---------

.. code-block:: ipython

  In [25]: from pyannote.core import Timeline, Segment

  In [26]: timeline = Timeline()
     ....: timeline.add(Segment(1, 5))
     ....: timeline.add(Segment(6, 8))
     ....: timeline.add(Segment(12, 18))
     ....: timeline.add(Segment(7, 20))
     ....: timeline

.. plot:: pyplots/timeline.py


Annotations
-----------


.. code-block:: ipython

    In [1]: from pyannote.core import Annotation, Segment

    In [6]: annotation = Annotation()
       ...: annotation[Segment(1, 5)] = 'Carol'
       ...: annotation[Segment(6, 8)] = 'Bob'
       ...: annotation[Segment(12, 18)] = 'Carol'
       ...: annotation[Segment(7, 20)] = 'Alice'
       ...: annotation

.. plot:: pyplots/annotation.py

"""

from collections.abc import Iterable
from itertools import cycle, groupby, product

import numpy as np

from .annotation import Annotation
from .feature import SlidingWindowFeature
from .segment import Segment, SlidingWindow
from .timeline import Timeline
from .utils.types import Label, LabelStyle, Resource

try:
    import matplotlib  # noqa: F401 -- probe whether the optional dependency imports
except ImportError:
    MATPLOTLIB_IS_AVAILABLE = False
else:
    MATPLOTLIB_IS_AVAILABLE = True

MATPLOTLIB_WARNING = (
    "Couldn't import matplotlib to render the vizualization "
    "for object {klass}. To enable, install the required dependencies "
    "with 'pip install pyannore.core[notebook]'"
)


class Notebook:
    def __init__(self):
        self.reset()

    def reset(self):
        from matplotlib.pyplot import get_cmap

        linewidth = [3, 1]
        linestyle = ["solid", "dashed", "dotted"]

        cm = get_cmap("Set1")
        colors = [cm(1.0 * i / 8) for i in range(9)]

        self._style_generator = cycle(product(linestyle, linewidth, colors))
        self._style: dict[Label | None, LabelStyle] = {
            None: ("solid", 1, (0.0, 0.0, 0.0))
        }
        del self.crop
        del self.width

    @property
    def crop(self):
        """The crop property."""
        return self._crop

    @crop.setter
    def crop(self, segment: Segment):
        self._crop = segment

    @crop.deleter
    def crop(self):
        self._crop = None

    @property
    def width(self):
        """The width property"""
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value

    @width.deleter
    def width(self):
        self._width = 20

    def __getitem__(self, label: Label) -> LabelStyle:
        """Get line style for a given label"""
        if label not in self._style:
            self._style[label] = next(self._style_generator)
        return self._style[label]

    def setup(self, ax=None, ylim=(0, 1), yaxis=False, time=True):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        ax.set_xlim(self.crop)
        if time:
            ax.set_xlabel("Time")
        else:
            ax.set_xticklabels([])
        ax.set_ylim(ylim)
        ax.axes.get_yaxis().set_visible(yaxis)
        return ax

    def draw_segment(self, ax, segment: Segment, y, label=None, boundaries=True):

        # do nothing if segment is empty
        if not segment:
            return

        linestyle, linewidth, color = self[label]

        # draw segment
        ax.hlines(
            y,
            segment.start,
            segment.end,
            color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
        )
        if boundaries:
            ax.vlines(
                segment.start, y + 0.05, y - 0.05, color, linewidth=1, linestyle="solid"
            )
            ax.vlines(
                segment.end, y + 0.05, y - 0.05, color, linewidth=1, linestyle="solid"
            )

        if label is None:
            return

    def get_y(self, segments: Iterable[Segment]) -> np.ndarray:
        """

        Parameters
        ----------
        segments : Iterable
            `Segment` iterable (sorted)

        Returns
        -------
        y : np.array
            y coordinates of each segment

        """

        # up_to stores the largest end time
        # displayed in each line (at the current iteration)
        # (at the beginning, there is only one empty line)
        up_to = [-np.inf]

        # y[k] indicates on which line to display kth segment
        y = []

        for segment in segments:
            # so far, we do not know which line to use
            found = False
            # try each line until we find one that is ok
            for i, u in enumerate(up_to):
                # if segment starts after the previous one
                # on the same line, then we add it to the line
                if segment.start >= u:
                    found = True
                    y.append(i)
                    up_to[i] = segment.end
                    break
            # in case we went out of lines, create a new one
            if not found:
                y.append(len(up_to))
                up_to.append(segment.end)

        # from line numbers to actual y coordinates
        y = 1.0 - 1.0 / (len(up_to) + 1) * (1 + np.array(y))

        return y

    def __call__(self, resource: Resource, time: bool = True, legend: bool = True):

        if isinstance(resource, Segment):
            self.plot_segment(resource, time=time)

        elif isinstance(resource, Timeline):
            self.plot_timeline(resource, time=time)

        elif isinstance(resource, Annotation):
            self.plot_annotation(resource, time=time, legend=legend)

        elif isinstance(resource, SlidingWindowFeature):
            self.plot_feature(resource, time=time)

    def plot_segment(self, segment, ax=None, time=True):

        if not self.crop:
            self.crop = segment

        ax = self.setup(ax=ax, time=time)
        self.draw_segment(ax, segment, 0.5)

    def plot_timeline(self, timeline: Timeline, ax=None, time=True):

        if not self.crop and timeline:
            self.crop = timeline.extent()

        cropped = timeline.crop(self.crop, mode="loose")

        ax = self.setup(ax=ax, time=time)

        for segment, y in zip(cropped, self.get_y(cropped)):
            self.draw_segment(ax, segment, y)

        # ax.set_aspect(3. / self.crop.duration)

    def plot_annotation(self, annotation: Annotation, ax=None, time=True, legend=True):

        if not self.crop:
            self.crop = annotation.get_timeline(copy=False).extent()

        cropped = annotation.crop(self.crop, mode="intersection")
        segments = [s for s, _ in cropped.itertracks()]

        ax = self.setup(ax=ax, time=time)

        for (segment, track, label), y in zip(
            cropped.itertracks(yield_label=True), self.get_y(segments)
        ):
            self.draw_segment(ax, segment, y, label=label)

        if legend:
            H, L = ax.get_legend_handles_labels()

            # corner case when no segment is visible
            if not H:
                return

            # this gets exactly one legend handle and one legend label per label
            # (avoids repeated legends for repeated tracks with same label)
            HL = groupby(
                sorted(zip(H, L), key=lambda h_l: h_l[1]), key=lambda h_l: h_l[1]
            )
            H, L = zip(*[(next(h_l)[0], l) for l, h_l in HL])
            ax.legend(
                H,
                L,
                bbox_to_anchor=(0, 1),
                loc=3,
                ncol=5,
                borderaxespad=0.0,
                frameon=False,
            )

    def plot_feature(
        self, feature: SlidingWindowFeature, ax=None, time=True, ylim=None
    ):

        if not self.crop:
            self.crop = feature.extent

        window = feature.sliding_window
        n, _dimension = feature.data.shape
        ((start, stop),) = window.crop(self.crop, mode="loose", return_ranges=True)
        xlim = (window[start].middle, window[stop].middle)

        start = max(0, start)
        stop = min(stop, n)
        t = window[0].middle + window.step * np.arange(start, stop)
        data = feature[start:stop]

        if ylim is None:
            m = np.nanmin(data)
            M = np.nanmax(data)
            ylim = (m - 0.1 * (M - m), M + 0.1 * (M - m))

        ax = self.setup(ax=ax, yaxis=False, ylim=ylim, time=time)
        ax.plot(t, data)
        ax.set_xlim(xlim)


notebook = Notebook()


def repr_segment(segment: Segment):
    """Get `png` data for `segment`"""
    import matplotlib.pyplot as plt
    from IPython.core.pylabtools import print_figure

    figsize = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = (notebook.width, 1)
    fig, ax = plt.subplots()
    notebook.plot_segment(segment, ax=ax)
    data = print_figure(fig, "png")
    plt.close(fig)
    plt.rcParams["figure.figsize"] = figsize
    return data


def repr_timeline(timeline: Timeline):
    """Get `png` data for `timeline`"""
    import matplotlib.pyplot as plt
    from IPython.core.pylabtools import print_figure

    figsize = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = (notebook.width, 1)
    fig, ax = plt.subplots()
    notebook.plot_timeline(timeline, ax=ax)
    data = print_figure(fig, "png")
    plt.close(fig)
    plt.rcParams["figure.figsize"] = figsize
    return data


def repr_annotation(annotation: Annotation):
    """Get `png` data for `annotation`"""
    import matplotlib.pyplot as plt
    from IPython.core.pylabtools import print_figure

    figsize = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = (notebook.width, 2)
    fig, ax = plt.subplots()
    notebook.plot_annotation(annotation, ax=ax)
    data = print_figure(fig, "png")
    plt.close(fig)
    plt.rcParams["figure.figsize"] = figsize
    return data


def repr_feature(feature: SlidingWindowFeature):
    """Get `png` data for `feature`"""
    import matplotlib.pyplot as plt
    from IPython.core.pylabtools import print_figure

    figsize = plt.rcParams["figure.figsize"]

    if feature.data.ndim == 2:
        plt.rcParams["figure.figsize"] = (notebook.width, 2)
        fig, ax = plt.subplots()
        notebook.plot_feature(feature, ax=ax)
        data = print_figure(fig, "png")
        plt.close(fig)

    elif feature.data.ndim == 3:
        num_chunks = len(feature)

        if notebook.crop is None:
            notebook.crop = Segment(
                start=feature.sliding_window.start,
                end=feature.sliding_window[num_chunks - 1].end,
            )
        else:
            feature = feature.crop(notebook.crop, mode="loose", return_data=False)

        num_overlap = (
            round(feature.sliding_window.duration // feature.sliding_window.step) + 1
        )

        num_overlap = min(num_chunks, num_overlap)

        plt.rcParams["figure.figsize"] = (notebook.width, 1.5 * num_overlap)

        fig, axes = plt.subplots(nrows=num_overlap, ncols=1, squeeze=False)
        axes = axes.flatten()
        mini, maxi = np.nanmin(feature.data), np.nanmax(feature.data)
        ylim = (mini - 0.2 * (maxi - mini), maxi + 0.2 * (maxi - mini))
        for c, (window, data) in enumerate(feature):
            ax = axes[c % num_overlap]
            step = duration = window.duration / len(data)
            frames = SlidingWindow(start=window.start, step=step, duration=duration)
            window_feature = SlidingWindowFeature(data, frames, labels=feature.labels)
            notebook.plot_feature(
                window_feature,
                ax=ax,
                time=c % num_overlap == (num_overlap - 1),
                ylim=ylim,
            )
            ax.set_prop_cycle(None)
        data = print_figure(fig, "png")
        plt.close(fig)

    plt.rcParams["figure.figsize"] = figsize
    return data
