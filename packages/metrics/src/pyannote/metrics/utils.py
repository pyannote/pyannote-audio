# The MIT License (MIT)

# Copyright (c) 2012-2019 CNRS

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

import warnings

from pyannote.core import Annotation, Segment, Timeline


class UEMSupportMixin:
    """Provides 'uemify' method with optional (à la NIST) collar"""

    def extrude(
        self,
        uem: Timeline,
        reference: Annotation,
        collar: float = 0.0,
        skip_overlap: bool = False,
    ) -> Timeline:
        """Extrude reference boundary collars from uem

        reference     |----|     |--------------|       |-------------|
        uem       |---------------------|    |-------------------------------|
        extruded  |--| |--| |---| |-----|    |-| |-----| |-----------| |-----|

        Parameters
        ----------
        uem : Timeline
            Evaluation map.
        reference : Annotation
            Reference annotation.
        collar : float, optional
            When provided, set the duration of collars centered around
            reference segment boundaries that are extruded from both reference
            and hypothesis. Defaults to 0. (i.e. no collar).
        skip_overlap : bool, optional
            Set to True to not evaluate overlap regions.
            Defaults to False (i.e. keep overlap regions).

        Returns
        -------
        extruded_uem : Timeline
        """

        if collar == 0.0 and not skip_overlap:
            return uem

        collars, overlap_regions = [], []

        # build list of collars if needed
        if collar > 0.0:
            # iterate over all segments in reference
            for segment in reference.itersegments():
                # add collar centered on start time
                t = segment.start
                collars.append(Segment(t - 0.5 * collar, t + 0.5 * collar))

                # add collar centered on end time
                t = segment.end
                collars.append(Segment(t - 0.5 * collar, t + 0.5 * collar))

        # build list of overlap regions if needed
        if skip_overlap:
            # iterate over pair of intersecting segments
            for (segment1, track1), (segment2, track2) in reference.co_iter(reference):
                if segment1 == segment2 and track1 == track2:
                    continue
                # add their intersection
                overlap_regions.append(segment1 & segment2)

        segments = collars + overlap_regions

        return Timeline(segments=segments).support().gaps(support=uem)

    def common_timeline(
        self, reference: Annotation, hypothesis: Annotation
    ) -> Timeline:
        """Return timeline common to both reference and hypothesis

        reference   |--------|    |------------|     |---------|         |----|
        hypothesis     |--------------| |------|   |----------------|
        timeline    |--|-----|----|---|-|------|   |-|---------|----|    |----|

        Parameters
        ----------
        reference : Annotation
        hypothesis : Annotation

        Returns
        -------
        timeline : Timeline
        """
        timeline = reference.get_timeline(copy=True)
        timeline.update(hypothesis.get_timeline(copy=False))
        return timeline.segmentation()

    def project(self, annotation: Annotation, timeline: Timeline) -> Annotation:
        """Project annotation onto timeline segments

        reference     |__A__|     |__B__|
                        |____C____|

        timeline    |---|---|---|   |---|

        projection  |_A_|_A_|_C_|   |_B_|
                        |_C_|

        Parameters
        ----------
        annotation : Annotation
        timeline : Timeline

        Returns
        -------
        projection : Annotation
        """
        projection = annotation.empty()
        timeline_ = annotation.get_timeline(copy=False)
        for segment_, segment in timeline_.co_iter(timeline):
            for track_ in annotation.get_tracks(segment_):
                track = projection.new_track(segment, candidate=track_)
                projection[segment, track] = annotation[segment_, track_]
        return projection

    def uemify(
        self,
        reference: Annotation,
        hypothesis: Annotation,
        uem: Timeline | None = None,
        collar: float = 0.0,
        skip_overlap: bool = False,
        returns_uem: bool = False,
        returns_timeline: bool = False,
    ) -> (
        tuple[Annotation, Annotation]
        | tuple[Annotation, Annotation, Timeline]
        | tuple[Annotation, Annotation, Timeline, Timeline]
    ):
        """Crop 'reference' and 'hypothesis' to 'uem' support

        Parameters
        ----------
        reference, hypothesis : Annotation
            Reference and hypothesis annotations.
        uem : Timeline, optional
            Evaluation map.
        collar : float, optional
            When provided, set the duration of collars centered around
            reference segment boundaries that are extruded from both reference
            and hypothesis. Defaults to 0. (i.e. no collar).
        skip_overlap : bool, optional
            Set to True to not evaluate overlap regions.
            Defaults to False (i.e. keep overlap regions).
        returns_uem : bool, optional
            Set to True to return extruded uem as well.
            Defaults to False (i.e. only return reference and hypothesis)
        returns_timeline : bool, optional
            Set to True to oversegment reference and hypothesis so that they
            share the same internal timeline.

        Returns
        -------
        reference, hypothesis : Annotation
            Extruded reference and hypothesis annotations
        uem : Timeline
            Extruded uem (returned only when 'returns_uem' is True)
        timeline : Timeline:
            Common timeline (returned only when 'returns_timeline' is True)
        """

        # when uem is not provided, use the union of reference and hypothesis
        # extents -- and warn the user about that.
        if uem is None:
            r_extent = reference.get_timeline().extent()
            h_extent = hypothesis.get_timeline().extent()
            extent = r_extent | h_extent
            uem = Timeline(segments=[extent] if extent else [], uri=reference.uri)
            warnings.warn(
                "'uem' was approximated by the union of 'reference' "
                "and 'hypothesis' extents."
            )

        # extrude collars (and overlap regions) from uem
        uem = self.extrude(uem, reference, collar=collar, skip_overlap=skip_overlap)

        # extrude regions outside of uem
        reference = reference.crop(uem, mode="intersection")
        hypothesis = hypothesis.crop(uem, mode="intersection")

        # project reference and hypothesis on common timeline
        if returns_timeline:
            timeline = self.common_timeline(reference, hypothesis)
            reference = self.project(reference, timeline)
            hypothesis = self.project(hypothesis, timeline)

        result = (reference, hypothesis)
        if returns_uem:
            result += (uem,)

        if returns_timeline:
            result += (timeline,)

        return result
