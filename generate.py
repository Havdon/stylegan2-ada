# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import argparse
import os
import pickle
import re

import numpy as np

from numpy import linalg
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib

#----------------------------------------------------------------------------

def generate_images(network_pkl, start_seed, truncation_psi, outdir, class_idx, dlatents_npz, diameter, frames):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    os.makedirs(outdir, exist_ok=True)

    # Render images for a given dlatent vector.
    if dlatents_npz is not None:
        print(f'Generating images from dlatents file "{dlatents_npz}"')
        dlatents = np.load(dlatents_npz)['dlatents']
        assert dlatents.shape[1:] == (18, 512) # [N, 18, 512]
        imgs = Gs.components.synthesis.run(dlatents, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
        for i, img in enumerate(imgs):
            fname = f'{outdir}/dlatent{i:02d}.png'
            print (f'Saved {fname}')
            PIL.Image.fromarray(img, 'RGB').save(fname)
        return

    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    label = np.zeros([1] + Gs.input_shapes[1][1:])
    if class_idx is not None:
        label[:, class_idx] = 1

    zs = get_circularloop(Gs, frames, diameter, start_seed)

    for seed_idx, z in enumerate(zs):
        print('Generating frame %d/%d ...' % (seed_idx, len(zs)))
        noise_rnd = np.random.RandomState(1)
        tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(f'{outdir}/frame_{seed_idx:04d}.png')


def get_circularloop(Gs, nf, d, seed):
    r = d/2
    if seed:
        np.random.RandomState(seed)

    zs = []

    rnd = np.random
    latents_a = rnd.randn(1, Gs.input_shape[1])
    latents_b = rnd.randn(1, Gs.input_shape[1])
    latents_c = rnd.randn(1, Gs.input_shape[1])
    latents = (latents_a, latents_b, latents_c)

    current_pos = 0.0
    step = 1./nf
    
    while(current_pos < 1.0):
        zs.append(circular_interpolation(r, latents, current_pos))
        current_pos += step
    return zs

def circular_interpolation(radius, latents_persistent, latents_interpolate):
    latents_a, latents_b, latents_c = latents_persistent

    latents_axis_x = (latents_a - latents_b).flatten() / linalg.norm(latents_a - latents_b)
    latents_axis_y = (latents_a - latents_c).flatten() / linalg.norm(latents_a - latents_c)

    latents_x = np.sin(np.pi * 2.0 * latents_interpolate) * radius
    latents_y = np.cos(np.pi * 2.0 * latents_interpolate) * radius

    latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
    return latents

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate curated MetFaces images without truncation (Fig.10 left)
  python %(prog)s --outdir=out --trunc=1 --seeds=85,265,297,849 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
  python %(prog)s --outdir=out --trunc=0.7 --seeds=600-605 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
  python %(prog)s --outdir=out --trunc=1 --seeds=0-35 --class=1 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/cifar10.pkl

  # Render image from projected latent vector
  python %(prog)s --outdir=out --dlatents=out/dlatents.npz \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate images using pretrained network pickle.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--start-seed', dest='start_seed', type=int, help='Starting seed for interpolation')
    g.add_argument('--dlatents', dest='dlatents_npz', help='Generate images for saved dlatents')
    parser.add_argument('--trunc', dest='truncation_psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser.add_argument('--class', dest='class_idx', type=int, help='Class label (default: unconditional)')
    parser.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')
    parser.add_argument('--diameter', type=int, default=50.0)
    parser.add_argument('--frames', type=int, default=5)

    args = parser.parse_args()
    generate_images(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
