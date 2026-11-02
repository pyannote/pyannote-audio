from functools import wraps

from pyannote.core import segment


def preserve_segment_state(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        prev_round_time = segment.AUTO_ROUND_TIME
        try:
            ret = f(*args, **kwargs)
        finally:
            segment.AUTO_ROUND_TIME = prev_round_time
        return ret

    return wrapper
