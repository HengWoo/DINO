def __getattr__(name):
    if name in ("SpatialPipeline", "SpatialResults"):
        from dino.spatial.spatial_pipeline import SpatialPipeline, SpatialResults

        globals()["SpatialPipeline"] = SpatialPipeline
        globals()["SpatialResults"] = SpatialResults
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["SpatialPipeline", "SpatialResults"]
