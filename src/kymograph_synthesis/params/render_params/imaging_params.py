from pydantic import Field
import microsim.schema as ms

# microsim params but exclude sample, ParticleSystem and RenderingParams are elsewhere
class ImagingParams(ms.Simulation):

    # use default factory to avoid same reference to labels list in each instance
    # (even though this is here as a dummy var)
    sample: ms.Sample = Field(
        default_factory=lambda: ms.Sample(labels=[]), exclude=True
    )

    output_space: ms.space.Space = Field(default=ms.DownscaledSpace(downscale=4))

    modality: ms.Modality = Field(default=ms.Widefield())

    detector: ms.detectors.Detector = Field(
        default=ms.CameraCCD(qe=1, read_noise=4, bit_depth=12, offset=100)
    )

    exposure_ms: float = 200