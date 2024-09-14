from functools import wraps


def make_fwd_ignore_context(obj):
    """Wraps object's forward method to ignore the context keyword."""
    original_forward = obj.forward

    @wraps(original_forward)
    def wrapped_forward(*args, context=None, **kwargs):
        return original_forward(*args, **kwargs)

    obj.forward = wrapped_forward
