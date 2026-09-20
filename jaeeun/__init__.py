"""Jaeeun's hair and outfit virtual-fitting package."""

__all__ = ["PipelineConfig", "VirtualFittingPipeline"]


def __getattr__(name: str):
	if name in __all__:
		from .pipeline import PipelineConfig, VirtualFittingPipeline

		return {"PipelineConfig": PipelineConfig, "VirtualFittingPipeline": VirtualFittingPipeline}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

