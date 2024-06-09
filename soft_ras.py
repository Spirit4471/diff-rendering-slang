import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from renderers.soft_ras import SoftRas
from resources.resource import spot_obj
from utils.util import wrap_float_tensor
from renderers.camera import PerspectiveCamera
from utils.vectors import Vector3
from renderers.transform import Transform, rotate_to_quaternion
from renderers.light import Light

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def single_forward(
    renderer, camera, face_vertices, transform, light, RenderParams
) -> torch.Tensor:
    # Create an instance of RenderParams
    # params = RenderParams(width, height, sigma)
    # Pass it to the forward function
    output = renderer.apply(camera, face_vertices, transform, light, RenderParams)
    return output


def main():
    renderer = SoftRas()
    # width = height = 512

    face_vertices = spot_obj.face_vertices
    print(face_vertices.shape)

    params = {
        "sigma": 1e-6,
        "epsilon": 1e-3,
        "gamma": 1e-4,
        "distance_epsilon": 1e-5,
        "fg_color": [0.7, 0.8, 0.9],
        "bg_color": [0.3, 0.2, 0.1],
    }

    camera = PerspectiveCamera(
        eye=Vector3(3.0, 0.0, 0.0),
        dir=Vector3(-1.0, 0.0, 0.0),
        up=Vector3(0.0, 1.0, 0.0),
        fov=60.0 / 180.0 * np.pi,
        near=0.1,
        far=100,
        width=512,
        height=384,
    )

    transform = Transform(
        rotation=rotate_to_quaternion(Vector3(1.0, 0.0, 0.0), 0.0 * np.pi),
        position=Vector3(0.0, 0.0, 0.0),
        scaling=Vector3(1.0, 1.5, 1.0),
    )
    light = Light(
        position=Vector3(10.0, 10.0, 10.0),
        color=Vector3(1.0, 1.0, 1.0)
    )
    output = single_forward(renderer, camera, face_vertices, transform, light, params)  
    print(output[0, 0, :])
    print(output.shape)

    output_image = output.cpu().detach().numpy()
    print("Output image shape before any transpose:", output_image.shape)
    
    if output_image.shape[0] == 1:
        output_image = output_image[0]
    if output_image.shape[0] == 3:
        output_image = output_image.transpose(1, 2, 0)
    print("Output image shape after correction:", output_image.shape)

    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
    output_image = np.flipud(output_image)
    if output_image.shape[2] == 1:
        output_image = np.repeat(output_image, 3, axis=2)
    elif output_image.shape[2] != 3 and output_image.shape[2] != 4:
        raise ValueError("Third dimension must be 1, 3 or 4")

    output_image_path = "rendered_image.png"
    plt.imsave(output_image_path, output_image)
    print(f"Image saved to {output_image_path}")

    plt.imshow(output.cpu().numpy(), origin="lower")
    plt.show()

if __name__ == "__main__":
    main()