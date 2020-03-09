# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
import numpy as np
import PIL.Image as Image
import dnnlib
import dnnlib.tflib as tflib
import pickle

class TattoModel():

    def __init__(self, options):

        seeds = np.random.randint(low=1, high=999999)
        random.seed(seeds)
        self.truncation = 1

        print(options['checkpoint'])
        with open(options['checkpoint'], 'rb') as file:
            G, D, self.Gs = pickle.load(file)
        self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]

        self.Gs_kwargs = dnnlib.EasyDict()
        self.Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.Gs_kwargs.randomize_noise = False
        if self.truncation is not None:
            self.Gs_kwargs.truncation_psi = self.truncation
        

    # Generate an image based on some text.
    def generate(self, z, truncation):

        z = z.reshape((1, 512))
        self.Gs_kwargs.truncation_psi = truncation
        tflib.set_vars({var: random.randn(*var.shape.as_list()) for var in self.noise_vars}) # [height, width]
        images = self.Gs.run(z, None, **self.Gs_kwargs) # [minibatch, height, width, channel]

        output = np.clip(images[0], 0, 255).astype(np.uint8)

        return output
