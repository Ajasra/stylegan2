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
    width = 512
    height = 512
    display = (width,height)
    
    senderName = "outputGAN"
    receiverName = "input"
    silent = True

    # window setup
    pygame.init() 
    pygame.display.set_caption(senderName)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # OpenGL init
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0,width,height,0,1,-1)
    glMatrixMode(GL_MODELVIEW)
    glDisable(GL_DEPTH_TEST)
    glClearColor(0.0,0.0,0.0,0.0)
    glEnable(GL_TEXTURE_2D)

    # init spout receiver
    spoutReceiverWidth = width
    spoutReceiverHeight = height
    # create spout receiver
    spoutReceiver = SpoutSDK.SpoutReceiver()
	# Its signature in c++ looks like this: bool pyCreateReceiver(const char* theName, unsigned int theWidth, unsigned int theHeight, bool bUseActive);
    spoutReceiver.pyCreateReceiver(receiverName,spoutReceiverWidth,spoutReceiverHeight, False)
    # create textures for spout receiver and spout sender 
    textureReceiveID = glGenTextures(1)
        
    # initalise receiver texture
    glBindTexture(GL_TEXTURE_2D, textureReceiveID)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    # copy data into texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, spoutReceiverWidth, spoutReceiverHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, None ) 
    glBindTexture(GL_TEXTURE_2D, 0)


    spoutSender = SpoutSDK.SpoutSender()
    spoutSenderWidth = width
    spoutSenderHeight = height
	# Its signature in c++ looks like this: bool CreateSender(const char *Sendername, unsigned int width, unsigned int height, DWORD dwFormat = 0);
    spoutSender.CreateSender(senderName, spoutSenderWidth, spoutSenderHeight, 0)
    # create textures for spout receiver and spout sender 
    textureSendID = glGenTextures(1)
    
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
    
    print()
    print('LISTENING')
    
    while (True):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sock.close()
                quit()
        
        #background = False
        
        '''
        d = sock.recvfrom(1024)
        data = d[0]
        addr = d[1]
        
        reply = data.decode('utf-8')
        print(reply)
        
        if reply == 'Exit':
            pygame.quit()
            sock.close()
            quit()
        '''

        # receive texture
        # Its signature in c++ looks like this: bool pyReceiveTexture(const char* theName, unsigned int theWidth, unsigned int theHeight, GLuint TextureID, GLuint TextureTarget, bool bInvert, GLuint HostFBO);
        spoutReceiver.pyReceiveTexture(receiverName, spoutReceiverWidth, spoutReceiverHeight, textureReceiveID.item(), GL_TEXTURE_2D, False, 0)

        glBindTexture(GL_TEXTURE_2D, textureReceiveID)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        # copy pixel byte array from received texture   
        data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, outputType=None)  #Using GL_RGB can use GL_RGBA 
        glBindTexture(GL_TEXTURE_2D, 0)
        # swap width and height data around due to oddness with glGetTextImage. http://permalink.gmane.org/gmane.comp.python.opengl.user/2423
        data.shape = (data.shape[1], data.shape[0], data.shape[2])

        update = data[1,0,0]

        if update > 200:
            z = data[0,:,0]
            z = z / 255.0 * 7 - 3.5
            z = np.array(z)
            z = np.expand_dims(z, axis=0)
            
            if truncation_psi is not None:
                Gs_kwargs.truncation_psi = truncation_psi

            #print('Generating image for seed %d ...' % (seed))
            rnd = np.random.RandomState(0)
            #z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
            #print(z)
            #print(z.shape)
            tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
            images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
            
            output = images[0]

            # setup the texture so we can load the output into it
            glBindTexture(GL_TEXTURE_2D, textureSendID);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            # copy output into texture
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, output )
                
            # setup window to draw to screen
            glActiveTexture(GL_TEXTURE0)
            # clean start
            glClear(GL_COLOR_BUFFER_BIT  | GL_DEPTH_BUFFER_BIT )
            # reset drawing perspective
            glLoadIdentity()
            # draw texture on screen
            glBegin(GL_QUADS)

            glTexCoord(0,0)        
            glVertex2f(0,0)

            glTexCoord(1,0)
            glVertex2f(width,0)

            glTexCoord(1,1)
            glVertex2f(width,height)

            glTexCoord(0,1)
            glVertex2f(0,height)

            glEnd()
            
            if silent:
                pygame.display.iconify()
                    
            # update window
            pygame.display.flip()        

            spoutSender.SendTexture(textureSendID.item(), GL_TEXTURE_2D, spoutSenderWidth, spoutSenderHeight, False, 0)
        
        #sock.sendto(msg_to_bytes(reply), (udp_ip, udp_port))
        
        
    sock.close()
    pygame.quit()
    quit()
        
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
