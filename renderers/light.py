from attr import dataclass

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.vectors import Vector3

@dataclass
class Light:
    position: Vector3
    color: Vector3

    def serialize(self):
        return {
            "position": self.position.to_list(),
            "color": self.color.to_list(),
        }

if __name__ == "__main__":
    import numpy as np
    from torch.autograd import Function
    import slangtorch

    light = Light(
        position=Vector3(10.0, 10.0, 10.0),
        color=Vector3(1.0, 1.0, 1.0)
    )

    light_module = slangtorch.loadModule("shaders/lighting.slang", verbose=True)

    class TestLight(Function):
        @staticmethod
        def forward(ctx, light: Light, normal: Vector3, viewDir: Vector3, fragPos: Vector3, shininess: float):
            ctx.save_for_backward(light)

            light_module.TestLight(
                light=light.serialize(),
                normal=normal.to_list(),
                viewDir=viewDir.to_list(),
                fragPos=fragPos.to_list(),
                shininess=shininess
            ).launchRaw(blockSize=(1, 1, 1), gridSize=(1, 1, 1))

    dispatcher = TestLight()
    dispatcher.apply(light, Vector3(0.0, 0.0, 1.0), Vector3(0.0, 0.0, 1.0), Vector3(0.0, 0.0, 0.0), 32.0)
