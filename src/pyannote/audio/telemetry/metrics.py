# Copyright (c) 2025 - Pruna AI GmbH. All rights reserved.
# Copyright (c) 2025 - pyannoteAI

"""OpenTelemetry metrics for tracking function usage.

Metrics can be enabled/disabled for the current python kernel:

>>> from pyannote.audio.telemetry.metrics import set_telemetry_metrics
>>> set_telemetry_metrics(True)   # enable
>>> set_telemetry_metrics(False)  # disable

To activate / deactivate globally:

>>> set_telemetry_metrics(True, save_choice_as_default=True)
"""

from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import yaml
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pyannote.audio import __version__

CONFIG_FILE = Path(__file__).parent / "config.yaml"

# Load initial configuration
with open(CONFIG_FILE) as config_file_stream:
    CONFIG = yaml.safe_load(config_file_stream)

OTLP_ENDPOINT = CONFIG["otlp_endpoint"]
OTLP_HEADERS = CONFIG.get("otlp_headers", {})
SESSION_ID = str(uuid4())
DEFAULT_LOG_LEVEL = CONFIG["telemetry_log_level"]

# Initialize metrics with basic setup
exporter = OTLPSpanExporter(endpoint=OTLP_ENDPOINT, headers=OTLP_HEADERS)
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(exporter))
tracer = provider.get_tracer(__name__)
atexit.register(provider.shutdown)


def track_model_init(model: "Model") -> None:
    """
    Increment the model initialization counter.

    Parameters
    ----------
    model : Model
        Instantiated model
    """
    if is_metrics_enabled():
        with tracer.start_as_current_span("oss-model-init") as span:
            span.set_attributes(
                {
                    "origin": getattr(model, "_otel_origin", "unknown"),
                    "version": __version__,
                    "session_id": SESSION_ID,
                },
            )


def track_pipeline_init(pipeline: "Pipeline") -> None:
    """
    Track pipeline initialization

    Parameters
    ----------
    pipeline : Pipeline
        Instantiated pipeline
    """
    if is_metrics_enabled():
        with tracer.start_as_current_span("oss-pipeline-init") as span:
            span.set_attributes(
                {
                    "origin": getattr(pipeline, "_otel_origin", "unknown"),
                    "name": getattr(pipeline, "_otel_name", "unknown"),
                    "version": __version__,
                    "session_id": SESSION_ID,
                },
            )


def track_pipeline_apply(
    pipeline: "Pipeline",
    file: "AudioFile",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    **kwargs: Any,
) -> None:
    """
    Track duration of files processed by pipelines

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline being applied.
    file : AudioFile
        The audio file being processed by the pipeline.
    num_speakers, min_speakers, max_speakers : int, optional
        Requested number of speakers in the audio file, if applicable.
    """

    if is_metrics_enabled():
        from pyannote.audio.core.io import Audio
        duration: float = Audio().get_duration(file)

        from pyannote.audio.pipelines.utils.diarization import set_num_speakers
        num_speakers, min_speakers, max_speakers = set_num_speakers(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        with tracer.start_as_current_span("oss-pipeline-apply") as span:
            span.set_attributes(
                {
                    "origin": getattr(pipeline, "_otel_origin", "unknown"),
                    "name": getattr(pipeline, "_otel_name", "unknown"),
                    "version": __version__,
                    "session_id": SESSION_ID,
                    "duration": duration,
                    "num_speakers": num_speakers,
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers if np.isfinite(max_speakers) else "inf",
                },
            )


# Set initial metrics enabled state in env var if not already set
if "PYANNOTE_METRICS_ENABLED" not in os.environ:
    os.environ["PYANNOTE_METRICS_ENABLED"] = str(CONFIG["metrics_enabled"]).lower()


def is_metrics_enabled() -> bool:
    """
    Check if metrics are enabled.

    Returns
    -------
    bool
        True if metrics are enabled, False otherwise.
    """

    if "PYANNOTE_METRICS_ENABLED" not in os.environ:
        raise ValueError("PYANNOTE_METRICS_ENABLED environment variable is not set.")

    return os.environ["PYANNOTE_METRICS_ENABLED"].lower() == "true"


def _save_metrics_config(enabled: bool) -> None:
    """
    Save metrics state to the configuration file.

    Parameters
    ----------
    enabled : bool
        Whether metrics should be enabled or disabled.
    """
    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    config["metrics_enabled"] = enabled

    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(config, f)


def set_telemetry_metrics(enabled: bool, save_choice_as_default: bool = False) -> None:
    """
    Enable or disable metrics globally.

    Parameters
    ----------
    enabled : bool
        Whether to enable or disable the metrics.
    save_choice_as_default : bool, optional
        If True, saves the state to the configuration file as the default value.
    """
    enabled = bool(enabled)
    os.environ["PYANNOTE_METRICS_ENABLED"] = str(enabled).lower()
    if save_choice_as_default:
        _save_metrics_config(enabled)


def set_opentelemetry_log_level(level: str) -> None:
    """
    Set the log level for OpenTelemetry loggers to control error visibility.

    This can be used to suppress error messages when telemetry fails.

    Parameters
    ----------
    level : str
        The log level to set. Must be one of: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.

        - 'DEBUG': Show all messages including detailed debugging information
        - 'INFO': Show informational messages, warnings and errors
        - 'WARNING': Show only warnings and errors (default)
        - 'ERROR': Show only errors
        - 'CRITICAL': Show only critical errors

    Raises
    ------
    ValueError
        If the provided level is not a valid logging level.

    Examples
    --------
    To suppress most error messages:

    >>> set_opentelemetry_log_level('ERROR')

    To show all messages:

    >>> set_opentelemetry_log_level('DEBUG')
    """
    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    level = level.upper()

    if level not in valid_levels:
        raise ValueError(f"Log level must be one of: {', '.join(valid_levels)}")

    # Configure OpenTelemetry loggers
    logging_level = getattr(logging, level)

    # Set log level for all OpenTelemetry loggers
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith("opentelemetry"):
            logging.getLogger(logger_name).setLevel(logging_level)


set_opentelemetry_log_level(DEFAULT_LOG_LEVEL)
