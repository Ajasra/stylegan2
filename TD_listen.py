# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import time
import socket
import cv2

sys.path.append('Library')

import pretrained_networks

import SpoutSDK
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

def msg_to_bytes(msg):
    return msg.encode('utf-8')


#----------------------------------------------------------------------------

def generate_images(network_pkl, truncation_psi):
    
    # spout setup
    width = 256 
    height = 256 
    display = (width,height)
    
    # init spout sender
    spoutSender = SpoutSDK.SpoutSender()
    spoutSenderWidth = width
    spoutSenderHeight = height
    
    spoutSender.CreateSender('Spout Python Sender', spoutSenderWidth, spoutSenderHeight, 0)
    
    # setup UDP
    udp_ip = "127.0.0.1"
    udp_port = 7000
    rec_port = 6000
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print('Setting up UDP on ip={} and port={}'.format(udp_ip, udp_port))
    except:
        print('Failed to create socket')
        sys.exit()
        
    try:
        sock.bind(('', rec_port))
        print('Listening on ip={} and port={}'.format(udp_ip, rec_port))
    except:
        print('Bind failed')
        sys.exit()
    
    starting_msg = "Ready"
    sock.sendto( msg_to_bytes(starting_msg), (udp_ip, udp_port))
    
    # load nmmetwork and prepare to generate
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    
    background = True
    print()
    print('LISTENING')
    seed = 1
    
    while background:
        
        
        #background = False
        
        d = sock.recvfrom(1024)
        data = d[0]
        addr = d[1]
        
        reply = data.decode('utf-8')
        print('received: {}'.format(reply))
        
        if reply == 'Exit':
            background = False
            
        reply = reply.split('_')
        if reply[0] == 'seed':
            seed = int(reply[1])
            
            if truncation_psi is not None:
                Gs_kwargs.truncation_psi = truncation_psi

            print('Generating image for seed %d ...' % (seed))
            rnd = np.random.RandomState(seed)
            z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
            tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
            images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
            res = PIL.Image.fromarray(images[0], 'RGB').save('results/result.png')
            '''res =  np.array(res) 
            res = res[:, :, ::-1].copy() 
            '''
            #cv2.imshow('result', images[0])
            #res.show()
        
        #sock.sendto(msg_to_bytes(reply), (udp_ip, udp_port))
        
        
    sock.close()
        
#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2))+1)
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate ffhq uncurated images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=6600-6625 --truncation-psi=0.5

  # Generate ffhq curated images (matches paper Figure 11)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=66,230,389,1518 --truncation-psi=1.0

  # Generate uncurated car images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=6000-6025 --truncation-psi=0.5

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style-mixing-example --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0
'''

#----------------------------------------------------------------------------

def main():
    
    print()
    print()
    print()
    print('GENERATOR STARTED')
    
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_images = subparsers.add_parser('generate-images', help='Generate images')
    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', default='results/002332.pkl')
    parser_generate_images.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_generate_images.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'generate-images': 'TD_listen.generate_images',
        'style-mixing-example': 'TD_listen.style_mixing_example'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
