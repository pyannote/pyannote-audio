# The MIT License (MIT)

# Copyright (c) 2017-2020 CNRS

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
# Herve BREDIN - http://herve.niderb.fr

"""
Evaluation

Usage:
  pyannote-metrics detection [--subset=<subset> --collar=<seconds> --skip-overlap] <database.task.protocol> <hypothesis.rttm>
  pyannote-metrics segmentation [--subset=<subset> --tolerance=<seconds>] <database.task.protocol> <hypothesis.rttm>
  pyannote-metrics overlap [--subset=<subset> --collar=<seconds>] <database.task.protocol> <hypothesis.rttm>
  pyannote-metrics diarization [--subset=<subset> --greedy --collar=<seconds> --skip-overlap] <database.task.protocol> <hypothesis.rttm>
  pyannote-metrics identification [--subset=<subset> --collar=<seconds> --skip-overlap] <database.task.protocol> <hypothesis.rttm>
  pyannote-metrics spotting [--subset=<subset> --latency=<seconds>... --filter=<expression>...] <database.task.protocol> <hypothesis.json>
  pyannote-metrics -h | --help
  pyannote-metrics --version

Options:
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  --subset=<subset>          Evaluated subset (train|developement|test) [default: test]
  --collar=<seconds>         Collar, in seconds [default: 0.0].
  --skip-overlap             Do not evaluate overlap regions.
  --tolerance=<seconds>      Tolerance, in seconds [default: 0.5].
  --greedy                   Use greedy diarization error rate.
  --latency=<seconds>        Evaluate with fixed latency.
  --filter=<expression>      Filter out target trials that do not match the
                             expression; e.g. use --filter="speech>10" to skip
                             target trials with less than 10s of speech from
                             the target.
  -h --help                  Show this screen.
  --version                  Show version.

All modes but "spotting" expect hypothesis using the RTTM file format.
RTTM files contain one line per speech turn, using the following convention:

SPEAKER {uri} 1 {start_time} {duration} <NA> <NA> {speaker_id} <NA> <NA>

    * uri: file identifier (as given by pyannote.database protocols)
    * start_time: speech turn start time in seconds
    * duration: speech turn duration in seconds
    * speaker_id: speaker identifier

"spotting" mode expects hypothesis using the following JSON file format.
It should contain a list of trial hypothesis, using the same trial order as
pyannote.database speaker spotting protocols (e.g. protocol.test_trial())

[
    {'uri': '<uri>', 'model_id': '<model_id>', 'scores': [[<t1>, <v1>], [<t2>, <v2>], ... [<tn>, <vn>]]},
    {'uri': '<uri>', 'model_id': '<model_id>', 'scores': [[<t1>, <v1>], [<t2>, <v2>], ... [<tn>, <vn>]]},
    {'uri': '<uri>', 'model_id': '<model_id>', 'scores': [[<t1>, <v1>], [<t2>, <v2>], ... [<tn>, <vn>]]},
    ...
    {'uri': '<uri>', 'model_id': '<model_id>', 'scores': [[<t1>, <v1>], [<t2>, <v2>], ... [<tn>, <vn>]]},
]

    * uri: file identifier (as given by pyannote.database protocols)
    * model_id: target identifier (as given by pyannote.database protocols)
    * [ti, vi]: [time, value] pair indicating that the system has output the
                score vi at time ti (e.g. [10.2, 0.2] means that the system
                gave a score of 0.2 at time 10.2s).

Calling "spotting" mode will create a bunch of files.
* <hypothesis.det.txt> contains DET curve using the following raw file format:
    <threshold> <fpr> <fnr>
* <hypothesis.lcy.txt> contains latency curves using this format:
    <threshold> <fpr> <fnr> <speaker_latency> <absolute_latency>

"""

import functools
import json
import sys
import warnings

import numpy as np
import pandas as pd

# command line parsing
from docopt import docopt
from pyannote.core import Annotation, Timeline

# evaluation protocols
from pyannote.database import get_protocol
from pyannote.database.util import get_annotated, load_rttm
from tabulate import tabulate

from pyannote.metrics.detection import (
    DetectionAccuracy,
    DetectionErrorRate,
    DetectionPrecision,
    DetectionRecall,
)
from pyannote.metrics.diarization import (
    DiarizationCoverage,
    DiarizationErrorRate,
    DiarizationPurity,
    GreedyDiarizationErrorRate,
)
from pyannote.metrics.identification import (
    IdentificationErrorRate,
    IdentificationPrecision,
    IdentificationRecall,
)
from pyannote.metrics.segmentation import (
    SegmentationCoverage,
    SegmentationPrecision,
    SegmentationPurity,
    SegmentationRecall,
)
from pyannote.metrics.spotting import LowLatencySpeakerSpotting

showwarning_orig = warnings.showwarning


def showwarning(message, category, *args, **kwargs):
    print(category.__name__ + ":", str(message))


warnings.showwarning = showwarning


def to_overlap(current_file: dict) -> Annotation:
    """Get overlapped speech reference annotation

    Parameters
    ----------
    current_file : `dict`
        File yielded by pyannote.database protocols.

    Returns
    -------
    overlap : `pyannote.core.Annotation`
        Overlapped speech reference.
    """

    reference = current_file["annotation"]
    overlap = Timeline(uri=reference.uri)
    for (s1, t1), (s2, t2) in reference.co_iter(reference):
        l1 = reference[s1, t1]
        l2 = reference[s2, t2]
        if l1 == l2:
            continue
        overlap.add(s1 & s2)
    return overlap.support().to_annotation()


def get_hypothesis(hypotheses, current_file):
    """Get hypothesis for given file

    Parameters
    ----------
    hypotheses : `dict`
        Speaker diarization hypothesis provided by `load_rttm`.
    current_file : `dict`
        File description as given by pyannote.database protocols.

    Returns
    -------
    hypothesis : `pyannote.core.Annotation`
        Hypothesis corresponding to `current_file`.
    """

    uri = current_file["uri"]

    if uri in hypotheses:
        return hypotheses[uri]

    # if the exact 'uri' is not available in hypothesis,
    # look for matching substring
    tmp_uri = [u for u in hypotheses if u in uri]

    # no matching speech turns. return empty annotation
    if len(tmp_uri) == 0:
        msg = f'Could not find hypothesis for file "{uri}"; assuming empty file.'
        warnings.warn(msg)
        return Annotation(uri=uri, modality="speaker")

    # exactly one matching file. return it
    if len(tmp_uri) == 1:
        hypothesis = hypotheses[tmp_uri[0]]
        hypothesis.uri = uri
        return hypothesis

    # more that one matching file. error.
    msg = f'Found too many hypotheses matching file "{uri}" ({tmp_uri}).'
    raise ValueError(msg.format(uri=uri, uris=tmp_uri))


def process_one(item, hypotheses=None, metrics=None):
    reference = item["annotation"]
    hypothesis = get_hypothesis(hypotheses, item)
    uem = get_annotated(item)
    return {
        key: metric(reference, hypothesis, uem=uem) for key, metric in metrics.items()
    }


def get_reports(protocol, subset, hypotheses, metrics):
    process = functools.partial(process_one, hypotheses=hypotheses, metrics=metrics)

    for item in getattr(protocol, subset)():
        process(item)

    return {key: metric.report(display=False) for key, metric in metrics.items()}


def reindex(report):
    """Reindex report so that 'TOTAL' is the last row"""
    index = list(report.index)
    i = index.index("TOTAL")
    return report.reindex(index[:i] + index[i + 1 :] + ["TOTAL"])


def detection(protocol, subset, hypotheses, collar=0.0, skip_overlap=False):
    options = {"collar": collar, "skip_overlap": skip_overlap}

    metrics = {
        "error": DetectionErrorRate(**options),
        "accuracy": DetectionAccuracy(**options),
        "precision": DetectionPrecision(**options),
        "recall": DetectionRecall(**options),
    }

    reports = get_reports(protocol, subset, hypotheses, metrics)

    report = reports["error"]
    accuracy = reports["accuracy"]
    precision = reports["precision"]
    recall = reports["recall"]

    report["accuracy", "%"] = accuracy[metrics["accuracy"].name, "%"]
    report["precision", "%"] = precision[metrics["precision"].name, "%"]
    report["recall", "%"] = recall[metrics["recall"].name, "%"]

    report = reindex(report)

    columns = list(report.columns)
    report = report[[columns[0]] + columns[-3:] + columns[1:-3]]

    summary = "Detection (collar = {:g} ms{})".format(
        1000 * collar, ", no overlap" if skip_overlap else ""
    )

    headers = (
        [summary]
        + [report.columns[i][0] for i in range(4)]
        + ["%" if c[1] == "%" else c[0] for c in report.columns[4:]]
    )

    print(
        tabulate(
            report,
            headers=headers,
            tablefmt="simple",
            floatfmt=".2f",
            numalign="decimal",
            stralign="left",
            missingval="",
            showindex="default",
            disable_numparse=False,
        )
    )


def segmentation(protocol, subset, hypotheses, tolerance=0.5):
    options = {"tolerance": tolerance}

    metrics = {
        "coverage": SegmentationCoverage(**options),
        "purity": SegmentationPurity(**options),
        "precision": SegmentationPrecision(**options),
        "recall": SegmentationRecall(**options),
    }

    reports = get_reports(protocol, subset, hypotheses, metrics)

    coverage = reports["coverage"]
    purity = reports["purity"]
    precision = reports["precision"]
    recall = reports["recall"]

    coverage = coverage[metrics["coverage"].name]
    purity = purity[metrics["purity"].name]
    precision = precision[metrics["precision"].name]
    recall = recall[metrics["recall"].name]

    report = pd.concat([coverage, purity, precision, recall], axis=1)
    report = reindex(report)

    headers = [
        f"Segmentation (tolerance = {1000 * tolerance:g} ms)",
        "coverage",
        "purity",
        "precision",
        "recall",
    ]
    print(
        tabulate(
            report,
            headers=headers,
            tablefmt="simple",
            floatfmt=".2f",
            numalign="decimal",
            stralign="left",
            missingval="",
            showindex="default",
            disable_numparse=False,
        )
    )


def diarization(
    protocol, subset, hypotheses, greedy=False, collar=0.0, skip_overlap=False
):
    options = {"collar": collar, "skip_overlap": skip_overlap}

    metrics = {
        "purity": DiarizationPurity(**options),
        "coverage": DiarizationCoverage(**options),
    }

    if greedy:
        metrics["error"] = GreedyDiarizationErrorRate(**options)
    else:
        metrics["error"] = DiarizationErrorRate(**options)

    reports = get_reports(protocol, subset, hypotheses, metrics)

    report = reports["error"]
    purity = reports["purity"]
    coverage = reports["coverage"]

    report["purity", "%"] = purity[metrics["purity"].name, "%"]
    report["coverage", "%"] = coverage[metrics["coverage"].name, "%"]

    columns = list(report.columns)
    report = report[[columns[0]] + columns[-2:] + columns[1:-2]]

    report = reindex(report)

    summary = "Diarization ({:s}collar = {:g} ms{})".format(
        "greedy, " if greedy else "",
        1000 * collar,
        ", no overlap" if skip_overlap else "",
    )

    headers = (
        [summary]
        + [report.columns[i][0] for i in range(3)]
        + ["%" if c[1] == "%" else c[0] for c in report.columns[3:]]
    )

    print(
        tabulate(
            report,
            headers=headers,
            tablefmt="simple",
            floatfmt=".2f",
            numalign="decimal",
            stralign="left",
            missingval="",
            showindex="default",
            disable_numparse=False,
        )
    )


def identification(protocol, subset, hypotheses, collar=0.0, skip_overlap=False):
    options = {"collar": collar, "skip_overlap": skip_overlap}

    metrics = {
        "error": IdentificationErrorRate(**options),
        "precision": IdentificationPrecision(**options),
        "recall": IdentificationRecall(**options),
    }

    reports = get_reports(protocol, subset, hypotheses, metrics)

    report = reports["error"]
    precision = reports["precision"]
    recall = reports["recall"]

    report["precision", "%"] = precision[metrics["precision"].name, "%"]
    report["recall", "%"] = recall[metrics["recall"].name, "%"]

    columns = list(report.columns)
    report = report[[columns[0]] + columns[-2:] + columns[1:-2]]

    report = reindex(report)

    summary = "Identification (collar = {:g} ms{})".format(
        1000 * collar, ", no overlap" if skip_overlap else ""
    )

    headers = (
        [summary]
        + [report.columns[i][0] for i in range(3)]
        + ["%" if c[1] == "%" else c[0] for c in report.columns[3:]]
    )

    print(
        tabulate(
            report,
            headers=headers,
            tablefmt="simple",
            floatfmt=".2f",
            numalign="decimal",
            stralign="left",
            missingval="",
            showindex="default",
            disable_numparse=False,
        )
    )


def spotting(protocol, subset, latencies, hypotheses, output_prefix, filter_func=None):
    if not latencies:
        Scores = []

    protocol.diarization = False

    trials = getattr(protocol, f"{subset}_trial")()
    for i, (current_trial, hypothesis) in enumerate(zip(trials, hypotheses)):
        # check trial/hypothesis target consistency
        if current_trial["model_id"] != hypothesis["model_id"]:
            msg = (
                "target mismatch in trial #{i} (found: {found}, should be: {should_be})"
            )
            raise ValueError(
                msg.format(
                    i=i,
                    found=hypothesis["model_id"],
                    should_be=current_trial["model_id"],
                )
            )

        # check trial/hypothesis file consistency
        if current_trial["uri"] != hypothesis["uri"]:
            msg = "file mismatch in trial #{i} (found: {found}, should be: {should_be})"
            raise ValueError(
                msg.format(i=i, found=hypothesis["uri"], should_be=current_trial["uri"])
            )

        # check at least one score is provided
        if not hypothesis["scores"]:
            msg = "empty list of scores in trial #{i}."
            raise ValueError(msg.format(i=i))

        timestamps, scores = zip(*hypothesis["scores"])

        if not latencies:
            Scores.append(scores)

        # check trial/hypothesis timerange consistency
        try_with = current_trial["try_with"]
        if not min(timestamps) >= try_with.start:
            msg = (
                "incorrect timestamp in trial #{i} "
                "(found: {found:g}, should be: >= {should_be:g})"
            )
            raise ValueError(
                msg.format(i=i, found=min(timestamps), should_be=try_with.start)
            )

    if not latencies:
        # estimate best set of thresholds
        scores = np.concatenate(Scores)
        epsilons = np.array(
            [n * 10 ** (-e) for e in range(4, 1, -1) for n in range(1, 10)]
        )
        percentile = np.concatenate(
            [epsilons, np.arange(0.1, 100.0, 0.1), 100 - epsilons[::-1]]
        )
        thresholds = np.percentile(scores, percentile)

    if not latencies:
        metric = LowLatencySpeakerSpotting(thresholds=thresholds)

    else:
        metric = LowLatencySpeakerSpotting(latencies=latencies)

    trials = getattr(protocol, f"{subset}_trial")()
    for i, (current_trial, hypothesis) in enumerate(zip(trials, hypotheses)):
        if filter_func is not None:
            speech = current_trial["reference"].duration()
            target_trial = speech > 0
            if target_trial and filter_func(speech):
                continue

        reference = current_trial["reference"]
        metric(reference, hypothesis["scores"])

    if not latencies:
        thresholds, fpr, fnr, eer, _ = metric.det_curve(return_latency=False)

        # save DET curve to hypothesis.det.txt
        det_path = f"{output_prefix}.det.txt"
        det_tmpl = "{t:.9f} {p:.9f} {n:.9f}\n"
        with open(det_path, mode="w") as fp:
            fp.write("# threshold false_positive_rate false_negative_rate\n")
            for t, p, n in zip(thresholds, fpr, fnr):
                line = det_tmpl.format(t=t, p=p, n=n)
                fp.write(line)

        print(f"> {det_path}")

        thresholds, fpr, fnr, _, _, speaker_lcy, absolute_lcy = metric.det_curve(
            return_latency=True
        )

        # save DET curve to hypothesis.det.txt
        lcy_path = f"{output_prefix}.lcy.txt"
        lcy_tmpl = "{t:.9f} {p:.9f} {n:.9f} {s:.6f} {a:.6f}\n"
        with open(lcy_path, mode="w") as fp:
            fp.write(
                "# threshold false_positive_rate false_negative_rate speaker_latency absolute_latency\n"
            )
            for t, p, n, s, a in zip(thresholds, fpr, fnr, speaker_lcy, absolute_lcy):
                if p == 1:
                    continue
                if np.isnan(s):
                    continue
                line = lcy_tmpl.format(t=t, p=p, n=n, s=s, a=a)
                fp.write(line)

        print(f"> {lcy_path}")

        print()
        print(f"EER% = {100 * eer:.2f}")

    else:
        results = metric.det_curve()
        logs = []
        for key in sorted(results):
            result = results[key]
            log = {"latency": key}
            for latency in latencies:
                thresholds, fpr, fnr, eer, _ = result[latency]
                # print('EER @ {latency}s = {eer:.2f}%'.format(latency=latency,
                #                                             eer=100 * eer))
                log[latency] = eer
                # save DET curve to hypothesis.det.{lcy}s.txt
                det_path = f"{output_prefix}.det.{key}.{latency:g}s.txt"
                det_tmpl = "{t:.9f} {p:.9f} {n:.9f}\n"
                with open(det_path, mode="w") as fp:
                    fp.write("# threshold false_positive_rate false_negative_rate\n")
                    for t, p, n in zip(thresholds, fpr, fnr):
                        line = det_tmpl.format(t=t, p=p, n=n)
                        fp.write(line)
            logs.append(log)
            det_path = f"{output_prefix}.det.{key}.XXs.txt"
            print(f"> {det_path}")

        print()
        df = 100 * pd.DataFrame.from_dict(logs).set_index("latency")[latencies]
        print(
            tabulate(
                df,
                tablefmt="simple",
                headers=["latency"] + [f"EER% @ {l:g}s" for l in latencies],
                floatfmt=".2f",
                numalign="decimal",
                stralign="left",
                missingval="",
                showindex="default",
                disable_numparse=False,
            )
        )


def main():
    arguments = docopt(__doc__, version="Evaluation")

    collar = float(arguments["--collar"])
    skip_overlap = arguments["--skip-overlap"]
    tolerance = float(arguments["--tolerance"])

    # protocol
    protocol_name = arguments["<database.task.protocol>"]

    preprocessors = {}
    if arguments["overlap"]:
        if skip_overlap:
            msg = (
                "Option --skip-overlap is not supported "
                "when evaluating overlapped speech detection."
            )
            sys.exit(msg)
        preprocessors = {"annotation": to_overlap}

    protocol = get_protocol(protocol_name, preprocessors=preprocessors)

    # subset (train, development, or test)
    subset = arguments["--subset"]

    if arguments["spotting"]:
        hypothesis_json = arguments["<hypothesis.json>"]
        with open(hypothesis_json, mode="r") as fp:
            hypotheses = json.load(fp)

        output_prefix = hypothesis_json[:-5]

        latencies = [float(l) for l in arguments["--latency"]]

        filters = arguments["--filter"]
        if filters:
            from sympy import lambdify, symbols, sympify

            speech = symbols("speech")
            filter_funcs = []
            filter_funcs = [
                lambdify([speech], sympify(expression)) for expression in filters
            ]
            filter_func = lambda speech: any(~func(speech) for func in filter_funcs)
        else:
            filter_func = None

        spotting(
            protocol,
            subset,
            latencies,
            hypotheses,
            output_prefix,
            filter_func=filter_func,
        )

        sys.exit(0)

    hypothesis_rttm = arguments["<hypothesis.rttm>"]

    try:
        hypotheses = load_rttm(hypothesis_rttm)

    except FileNotFoundError:
        msg = f"Could not find file {hypothesis_rttm}."
        sys.exit(msg)

    except (OSError, ValueError, KeyError, IndexError):
        msg = (
            f"Failed to load {hypothesis_rttm}, please check its format "
            f"(only RTTM files are supported)."
        )
        sys.exit(msg)

    if arguments["detection"]:
        detection(
            protocol, subset, hypotheses, collar=collar, skip_overlap=skip_overlap
        )

    if arguments["overlap"]:
        detection(
            protocol, subset, hypotheses, collar=collar, skip_overlap=skip_overlap
        )

    if arguments["segmentation"]:
        segmentation(protocol, subset, hypotheses, tolerance=tolerance)

    if arguments["diarization"]:
        greedy = arguments["--greedy"]
        diarization(
            protocol,
            subset,
            hypotheses,
            greedy=greedy,
            collar=collar,
            skip_overlap=skip_overlap,
        )

    if arguments["identification"]:
        identification(
            protocol, subset, hypotheses, collar=collar, skip_overlap=skip_overlap
        )


if __name__ == "__main__":
    main()
