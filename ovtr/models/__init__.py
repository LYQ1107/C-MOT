def build(*args, **kwargs):
    # Lazy import keeps light-weight contract tests usable without CLIP while
    # preserving the upstream ``from models import build`` entry point.
    from .ovtr import build as _build
    return _build(*args, **kwargs)


def build_model(args, cfgs):
    return build(args, cfgs)

