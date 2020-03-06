#!/usr/bin/env python
r'''Tool to stitch videos from the abyss rig.

HERO 7 wide FOV: Vertical: 94.4, Horizontal 122.6

Example:

python  abyss2pano.py --campaign Bimini-2019 --date 2019-08-24 --rig 1 --out_frame_rate 30 --stitch cube --ext LRV


'''
from __future__ import division

import os
import re
import time
import subprocess
import argparse
from glob import glob
from datetime import datetime
from itertools import zip_longest

import skvideo.io
import numpy as np
import skimage
from spatialmedia import metadata_utils


#skvideo._FFMPEG_SUPPORTED_DECODERS.append(b'lrv')

class Console(object):
    def __init__(self):
        self.log = []

    def append(self, text):
        print(text.encode('utf-8'))
        self.log.append(text)


class VideoReaderWithLRV(skvideo.io.FFmpegReader):
    def _getSupportedDecoders(self, *args, **kwargs):
        supported_decoders = super(VideoReaderWithLRV, self)._getSupportedDecoders(*args, **kwargs)
        supported_decoders.append(b'.lrv')
        return supported_decoders

def gopro_file_name_parser(fn):
    '''Returns the video number, chapter number, encoding and extension of a gopro file name.'''
    pattern = re.compile('(?P<path>.+)(?P<enc>G\w)(?P<chapter>\d\d)(?P<video>\d\d\d\d)(?P<ext>\.\w\w\w)$')
    ma = pattern.match(fn)
    try:
        chapter = int(ma.group('chapter'))
        video = int(ma.group('video'))
        encoding = ma.group('enc')
        ext = ma.group('ext')
        path = ma.group('path')
    except AttributeError as err:
        print('Error parsing file name {}'.format(fn))
        print('The structure of the path/(encoding)(chapter)(video)(extension)')
        raise err
    return video, chapter, encoding, ext, path


def is_cuttingboard(fn, size_th):
    """Decides if fn belongs to the cutting board of if it contains information.

    The criterion to decide if a file is part of a recording is the following.
    If a video is part of a recording it has to have a size of aprox 4GB
    (maximum size of a file in the FAT system). But, if the file is the last
    one of serie then it migh be less than 4GB. So, the criterion is to check
    that the file's size is bigger than 4GB or that it is the last one and not
    the first one.

    The size threshold is a parameter in case the user wants to stitch low resoltuion videos which are less than 4GB.
    The GoPro7 black yelds 140MB LRV (Low Resolution Videos).

    Inputs:
    fn: (str) file name
    size_th: (float) size threshold in GB. Common cases are 3, for MP4 files and 0.1 for LRV.
    Outputs:
    out: (bool) True if fn does not contain information.
    """

    video, chapter, encoding, ext, path = gopro_file_name_parser(fn)
    is_first = chapter == 1
    next_chap = str(chapter + 1).zfill(2)
    next_fn = encoding + next_chap + str(video).zfill(4) + ext
    next_path = os.path.join(path, next_fn)
    is_last = os.path.exists(next_path)
    size_gb = os.stat(fn).st_size/10**9

    return size_gb>size_th or (is_last and not is_first)


def abyss2sph(full_frame_size, frame_shape, L):
    '''Abyss rig fotage to 360 video that looks like a sphere.

    Inputs:
    full_frame_sieze: (h, w, c) final frame size.
    frame_shape: (h, w) gopro videos shape
    L: distance between sensors in pixels

    Output:
    full_frame: np.array with the final frame size filled with zeros.
    ii: i coordinates in the input frames.
    jj: j coordinates in the input frames.
    theta_ii: i coordinates in final frame
    phi_jj: j coordinates in final frame

    Usage:
    full_frame, ii, jj, theta_ii, phi_jj = abyss2esf((2*resolution_90, 4*resolution_90, 3), (1440, 1920, 3), 4/3)
    full_frame[theta_ii[0], phi_jj[0], :] = frame1[ii[0], jj[0], :]
    full_frame[theta_ii[1], phi_jj[1], :] = frame2[ii[1], jj[1], :]
    full_frame[theta_ii[2], phi_jj[2], :] = frame3[ii[2], jj[2], :]
    full_frame[theta_ii[3], phi_jj[3], :] = frame4[ii[3], jj[3], :]
    full_frame[theta_ii[4], phi_jj[4], :] = frame5[ii[4], jj[4], :]
    full_frame[theta_ii[5], phi_jj[5], :] = frame6[ii[5], jj[5], :]
    '''

    wa = L*np.pi/2
    #sa = 3*wa/4
    sa = L*np.pi/2
    full_frame_size = list(full_frame_size)
    if len(full_frame_size)<3:
        full_frame_size.append(3)
    full_frame = np.zeros(full_frame_size).astype(np.int32)
    phi = np.linspace(0, 2*np.pi, full_frame.shape[1])
    theta = np.linspace(0, np.pi, full_frame.shape[0])
    theta, phi = np.meshgrid(theta, phi, indexing='ij')
    theta_i = np.round((full_frame.shape[0]-1)*theta/(np.pi)).astype(int)
    phi_j = np.round((full_frame.shape[1]-1)*phi/(2*np.pi)).astype(int)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # 5 panel
    alpha = np.pi/4
    u = np.cos(alpha) * y - np.sin(alpha) * z
    v = np.sin(alpha) * y + np.cos(alpha) * z
    u_theta = np.arccos(v)
    v_phi = np.arctan2(u, x)
    mask = (u_theta>=np.pi/2 - sa/2) & (u_theta<=np.pi/2 + sa/2) & (v_phi>=-wa/2) & (v_phi<=wa/2)
    u_norm = (u_theta - (np.pi/2 - sa/2))/sa
    v_norm = (v_phi + wa/2)/wa
    theta_i5 = theta_i[mask]
    phi_j5 = phi_j[mask]
    i5 = np.round((frame_shape[0]- 1) * u_norm[mask]).astype(int)
    j5 = np.round((frame_shape[1] - 1) * v_norm[mask]).astype(int)

    # 2 panel
    v_phi[v_phi<0] = v_phi[v_phi<0]+2*np.pi
    mask = (u_theta>=np.pi/2 - sa/2) & (u_theta<=np.pi/2 + sa/2) & (v_phi>=np.pi-wa/2) & (v_phi<=np.pi+wa/2)
    u_norm = (u_theta - (np.pi/2 - sa/2))/sa
    v_norm = (v_phi - (np.pi-wa/2))/wa
    theta_i2 = theta_i[mask]
    phi_j2 = phi_j[mask]
    i2 = np.round((frame_shape[0]- 1) * u_norm[mask]).astype(int)
    j2 = np.round((frame_shape[1] - 1) * v_norm[mask]).astype(int)

    # 6 panel
    mask = (u_theta>=np.pi/2 - wa/2) & (u_theta<=np.pi/2 + wa/2) & (v_phi>=3*np.pi/2-sa/2) & (v_phi<=3*np.pi/2+sa/2)
    u_norm = (u_theta - (np.pi/2 - wa/2))/wa
    v_norm = (v_phi - (3*np.pi/2 - sa/2))/sa
    theta_i6 = theta_i[mask]
    phi_j6 = phi_j[mask]
    i6 = np.round((frame_shape[0]- 1) * v_norm[mask]).astype(int)
    j6 = np.round((frame_shape[1] - 1) * (1-u_norm[mask])).astype(int)

    #Frame 3
    mask = (u_theta>=np.pi/2 - wa/2) & (u_theta<=np.pi/2 + wa/2) & (v_phi>=np.pi/2-sa/2) & (v_phi<=np.pi/2+sa/2)
    u_norm = (u_theta - (np.pi/2 - wa/2))/wa
    v_norm = (v_phi - (np.pi/2-sa/2))/sa

    theta_i3 = theta_i[mask]
    phi_j3 = phi_j[mask]
    i3 = np.round((frame_shape[0] - 1) * (1-v_norm[mask])).astype(int)
    j3 = np.round((frame_shape[1] - 1) * u_norm[mask]).astype(int)

    #Frame 1
    alpha = -np.pi/4
    u = np.cos(alpha) * y - np.sin(alpha) * z
    v = np.sin(alpha) * y + np.cos(alpha) * z
    u_theta = np.arccos(v)
    v_phi = np.arctan2(u, x)

    mask = (u_theta>=np.pi/2 - sa/2) & (u_theta<=np.pi/2 + sa/2) & (v_phi>=np.pi/2-wa/2) & (v_phi<=np.pi/2+wa/2)
    u_norm = (u_theta - (np.pi/2 - sa/2))/sa
    v_norm = (v_phi - (np.pi/2-wa/2))/wa
    theta_i1 = theta_i[mask]
    phi_j1 = phi_j[mask]
    i1 = np.round((frame_shape[0] - 1) * (1-u_norm[mask])).astype(int)
    j1 = np.round((frame_shape[1] - 1) * (1-v_norm[mask])).astype(int)

    # Frame 4
    v_phi[v_phi<0] = v_phi[v_phi<0] + 2*np.pi
    mask = (u_theta>=np.pi/2 - sa/2) & (u_theta<=np.pi/2 + sa/2) & (v_phi>=3*np.pi/2-wa/2) & (v_phi<=3*np.pi/2+wa/2)
    u_norm = (u_theta - (np.pi/2 - sa/2))/sa
    v_norm = (v_phi - (3*np.pi/2-wa/2))/wa
    theta_i4 = theta_i[mask]
    phi_j4 = phi_j[mask]
    i4 = np.round((frame_shape[0] - 1) * (1-u_norm[mask])).astype(int)
    j4 = np.round((frame_shape[1] - 1) * (1-v_norm[mask])).astype(int)

    ii = (i1, i2, i3, i4, i5, i6)
    jj = (j1, j2, j3, j4, j5, j6)
    theta_ii = (theta_i1, theta_i2, theta_i3, theta_i4, theta_i5, theta_i6)
    phi_jj = (phi_j1, phi_j2, phi_j3, phi_j4, phi_j5, phi_j6)
    return full_frame, ii, jj, theta_ii, phi_jj



def abys2cube(full_frame_size, frame_shape, L):
    '''Abyss rig fotage to 360 cube video.

    Inputs:
    full_frame_sieze: (h, w, c) final frame size.
    frame_shape: (h, w) gopro videos shape
    L: distance between sensors in pixels

    Output:
    full_frame: np.array with the final frame size filled with zeros.
    ii: i coordinates in the input frames.
    jj: j coordinates in the input frames.
    theta_ii: i coordinates in final frame
    phi_jj: j coordinates in final frame

    Usage:
    resolution_90 = 1440
    # HERO 7 FOV = 94.4 122.6
    full_frame_size = (2*resolution_90, 4*resolution_90, 3)
    frame_shape = (1440, 1920, 3)
    full_frame, ii, jj, theta_ii, phi_jj = abys2cube((full_frame_size, frame_shape,, 1/3+1/4)
    full_frame[theta_ii[0], phi_jj[0], :] = frame1[ii[0], jj[0], :]
    full_frame[theta_ii[1], phi_jj[1], :] = frame2[ii[1], jj[1], :]
    full_frame[theta_ii[2], phi_jj[2], :] = frame3[ii[2], jj[2], :]
    full_frame[theta_ii[3], phi_jj[3], :] = frame4[ii[3], jj[3], :]
    full_frame[theta_ii[4], phi_jj[4], :] = frame5[ii[4], jj[4], :]
    full_frame[theta_ii[5], phi_jj[5], :] = frame6[ii[5], jj[5], :]
    '''

    r = L
    full_frame_size = list(full_frame_size)
    if len(full_frame_size)<3:
        full_frame_size.append(3)
    full_frame = np.zeros(full_frame_size).astype(np.int32)
    phi = np.linspace(0, 2*np.pi, full_frame.shape[1])
    theta = np.linspace(0, np.pi, full_frame.shape[0])
    theta, phi = np.meshgrid(theta, phi, indexing='ij')
    theta_i = np.round((full_frame.shape[0]-1)*theta/(np.pi)).astype(int)
    phi_j = np.round((full_frame.shape[1]-1)*phi/(2*np.pi)).astype(int)

    # Frame 5
    r = L / (np.sin(theta) * np.cos(phi))
    y = L *  np.tan(phi)
    z = L/ (np.tan(theta) * np.cos(phi))
    alpha = np.pi/4
    u = np.cos(alpha) * y - np.sin(alpha) * z
    v = np.sin(alpha) * y + np.cos(alpha) * z
    mask = (u>=-2/3) & (u<=2/3) & (v>=-1/2) & (v<=1/2) & (r>0)
    u = u[mask]
    v = v[mask]
    theta_i5 = theta_i[mask]
    phi_j5 = phi_j[mask]
    u_norm = (u+2/3)/(4/3)
    v_norm = v + 1/2
    i5 = np.round((frame_shape[0]- 1) * (1 - v_norm)).astype(int)
    j5 = np.round((frame_shape[1] - 1) * u_norm).astype(int)


    #Frame 2
    r = L / (np.sin(theta) * np.cos(phi))
    y = L *  np.tan(phi)
    z = L/ (np.tan(theta) * np.cos(phi))
    alpha = -np.pi/4
    u = np.cos(alpha) * y - np.sin(alpha) * z
    v = np.sin(alpha) * y + np.cos(alpha) * z
    mask = (u>=-1/2) & (u<=1/2) & (v>=-2/3) & (v<=2/3) & (r<0)
    u = u[mask]
    v = v[mask]
    theta_i2 = theta_i[mask]
    phi_j2 = phi_j[mask]
    v_norm = (v+2/3)/(4/3)
    u_norm = u + 1/2
    i2 = np.round((frame_shape[0]- 1) * u_norm).astype(int)
    j2 = np.round((frame_shape[1] - 1) * (1 - v_norm)).astype(int)

    #Frame 6

    # using the plane equation in polar coordinates:
    r = np.sqrt(2) * L / (np.cos(theta) - np.sin(theta) * np.sin(phi))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    y_max = (2/3 - L) / np.sqrt(2)
    y_min = -(2/3 + L) / np.sqrt(2)
    mask = (y>=y_min) & (y<=y_max) & (x>=-1/2) & (x<=1/2) & (r>0)

    x = x[mask]
    y = y[mask]
    z = z[mask]
    u = np.sqrt((y_max-y)**2+(-y_min-z)**2)
    theta_i6 = theta_i[mask]
    phi_j6 = phi_j[mask]

    u_norm = u/(4/3)
    x_norm = x + 1/2
    i6 = np.round((frame_shape[0] - 1) * x_norm).astype(int)
    j6 = np.round((frame_shape[1] - 1) * (1 - u_norm)).astype(int)

    #Frame 3
    # using the plane equation in polar coordinates:
    r = np.sqrt(2) * L / (np.cos(theta) - np.sin(theta) * np.sin(phi))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    y_max = (2/3 - L) / np.sqrt(2)
    y_min = -(2/3 + L) / np.sqrt(2)
    mask = (y>=y_min) & (y<=y_max) & (x>=-1/2) & (x<=1/2) & (r<0)

    x = x[mask]
    y = y[mask]
    z = z[mask]
    u = np.sqrt((y_max-y)**2+(-y_min-z)**2)
    theta_i3 = theta_i[mask]
    phi_j3 = phi_j[mask]

    u_norm = u/(4/3)
    x_norm = x + 1/2
    i3 = np.round((frame_shape[0] - 1) * (1 - x_norm)).astype(int)
    j3 = np.round((frame_shape[1] - 1) * (1 - u_norm)).astype(int)

    #Frame 1
    # using the plane equation in polar coordinates:
    r = np.sqrt(2) * L / (np.cos(theta) + np.sin(theta) * np.sin(phi))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    y_max = (L + 1/2) / np.sqrt(2)
    y_min = (L - 1/2) / np.sqrt(2)
    mask = (y>=y_min) & (y<=y_max) & (x>=-2/3) & (x<=2/3) & (r>0)

    x = x[mask]
    y = y[mask]
    z = z[mask]
    u = np.sqrt((y_max-y)**2+(y_min-z)**2)
    theta_i1 = theta_i[mask]
    phi_j1 = phi_j[mask]

    x_norm = (x+2/3)/(4/3)
    u_norm = u
    i1 = np.round((frame_shape[0] - 1) * u_norm).astype(int)
    j1 = np.round((frame_shape[1] - 1) * x_norm).astype(int)


    # Frame 4
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    y_max = (1/2 + L) / np.sqrt(2)
    y_min = (L - 1/2) / np.sqrt(2)
    mask = (y>=y_min) & (y<=y_max) & (x>=-2/3) & (x<=2/3) & (r<0)

    x = x[mask]
    y = y[mask]
    z = z[mask]
    u = np.sqrt((y_max-y)**2+(y_min-z)**2)
    theta_i4 = theta_i[mask]
    phi_j4 = phi_j[mask]

    u_norm = u
    x_norm = (x + 2/3)/(4/3)
    i4 = np.round((frame_shape[0] - 1) * (1-u_norm)).astype(int)
    j4 = np.round((frame_shape[1] - 1) * x_norm).astype(int)
    ii = (i1, i2, i3, i4, i5, i6)
    jj = (j1, j2, j3, j4, j5, j6)
    theta_ii = (theta_i1, theta_i2, theta_i3, theta_i4, theta_i5, theta_i6)
    phi_jj = (phi_j1, phi_j2, phi_j3, phi_j4, phi_j5, phi_j6)
    return full_frame, ii, jj, theta_ii, phi_jj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--campaign', type=os.path.abspath, help=('path to campaign folder. Can be either relative or absolute '
                                                                  '(e.g. Bimini-2019).'))
    parser.add_argument('--date', type=lambda x: datetime.strptime(x,'%Y-%m-%d'),
                        help='date with the files to stitch in yyyy-mm-dd format (e.g. 2019-08-26).')
    parser.add_argument('--rig', default='1', choices=['1', '2'], help='rig number (1 or 2)')
    parser.add_argument('--out_frame_rate', default=30, type=int, help='frame rate of the output video')
    parser.add_argument('--stitch', default='cube', choices=['cube', 'sphere'],
                        help='type of stitching. cube or sphere')
    parser.add_argument('--ext', default='MP4', choices=['MP4', 'LRV'],
                        help=('extension of the files to stitch. The MP4 files are the full resolution ones, '
                              'and the LRV files are low resolution. Use LRV for faster stitching.'))
    
    args = parser.parse_args()

    date =  args.date.strftime('%Y-%m-%d')
    rig = args.rig
    out_frame_rate = args.out_frame_rate

    if args.stitch == 'cube':
        stitch_folder = 'dirty_cube'
    elif args.stitch == 'sphere':
        stitch_folder = 'dirty_sphere'

    root = os.path.join(args.campaign, date)
    if not os.path.isdir(root):
        raise Exception('{} is not a folder'.format(root))

    dst_folder = os.path.join(args.campaign, 'renders', stitch_folder)
    if not os.path.isdir(dst_folder):
        raise Exception('{} is not a folder'.format(dst_folder))


    cameras = sorted(glob(os.path.join(root, '*[rR]ig{}/*'.format(rig))))
    files_in_cameras = [sorted(glob(os.path.join(camera, '*.'+args.ext))) for camera in cameras]
    if args.ext == 'MP4':
        size_th = 3
    elif args.ext == 'LRV':
        size_th = 0.1
    files_in_cameras = [[x for x in fns if is_cuttingboard(x, size_th)] for fns in files_in_cameras]
    files_in_cameras = [sorted(fns, key=lambda x: gopro_file_name_parser(x)[:2]) for fns in files_in_cameras]
    files_per_chunk = [fns for fns in zip(*files_in_cameras)]
    videos_per_chunk = [[VideoReaderWithLRV(fn) for fn in fns] for fns in zip(*files_in_cameras)]

    metadata = skvideo.io.ffprobe(files_per_chunk[0][0])['video']
    frame_rate = metadata['@avg_frame_rate'].split('/')
    frame_rate = round(int(frame_rate[0]) / int(frame_rate[1]))
    frame_shape = (int(metadata["@height"]), int(metadata["@width"]), 3)
    resolution_90 = int(90*int(metadata["@height"])/94.4)
    full_frame_size = (2*resolution_90, 4*resolution_90, 3)
    if frame_rate < out_frame_rate:
        print('You set out_frame_rate to {} but input frame rate is {}. Changing output frame rate to {}'.format(out_frame_rate, frame_rate, out_frame_rate))
    out_frame_rate = min(frame_rate, out_frame_rate)


    out_file_format = os.path.join(dst_folder, "{}_rig{}_{}.MP4".format(date, rig,'{}'))
    out_file_sound_format = os.path.join(dst_folder, "{}_{}_r{}c1_audio.MP4".format(date, '{}', rig))
    out_file_sound_360_format = os.path.join(dst_folder, "{}_{}_r{}c1_audio_injected.{}".format(date, '{}', rig, args.ext))

    if args.stitch == 'cube':
        full_frame, ii, jj, theta_ii, phi_jj = abys2cube(full_frame_size, frame_shape, 1/3+1/4)
    elif args.stitch == 'sphere':
        full_frame, ii, jj, theta_ii, phi_jj = abyss2sph(full_frame_size, frame_shape, 4/3)

    console = Console()
    metadata = metadata_utils.Metadata()
    metadata.video = metadata_utils.generate_spherical_xml(stereo=None)

    for v_num, videos in enumerate(videos_per_chunk):
        audio_file = files_per_chunk[v_num][0]
        print('videos {}'.format(v_num))
        inputdict = {'-r': str(out_frame_rate/1.001)} #, '-ss': '2'
        outputdict = {"-c:v": "libx264", "-preset": "fast", "-crf": "17",
                      "-maxrate": "200M", "-bufsize": "25M", "-x264-params": "mvrange=511",
                      "-vf": "scale={}x{}:out_range=full:out_color_matrix=bt709".format(resolution_90*4, resolution_90*2),
                      "-pix_fmt": "yuv420p", "-r":  str(int(out_frame_rate)/1.001), "-movflags": "faststart"}
        out_file = out_file_format.format(v_num)
        writer = skvideo.io.FFmpegWriter(out_file, inputdict=inputdict, outputdict=outputdict)
        t0 = time.time()
        for i, (frame1, frame2, frame3, frame4, frame5, frame6) in enumerate(zip_longest(*videos, fillvalue=np.zeros(frame_shape))):
            if i % (frame_rate * 60) == 0:
                print('minutes {}'.format(int(i/frame_rate/60)))
                print('time {}'.format(time.time()-t0))
                t0 = time.time()
            if i % round(frame_rate/out_frame_rate) != 0:
                continue
            # there was a camera upside down...
            # TODO: load data stream and check camera orientation.
            # if v_num % 2 == 1:
            #     frame1 = frame1[::-1, ::-1, :]
            full_frame[theta_ii[0], phi_jj[0], :] = frame1[ii[0], jj[0], :]
            full_frame[theta_ii[1], phi_jj[1], :] = frame2[ii[1], jj[1], :]
            full_frame[theta_ii[2], phi_jj[2], :] = frame3[ii[2], jj[2], :]
            full_frame[theta_ii[3], phi_jj[3], :] = frame4[ii[3], jj[3], :]
            full_frame[theta_ii[4], phi_jj[4], :] = frame5[ii[4], jj[4], :]
            full_frame[theta_ii[5], phi_jj[5], :] = frame6[ii[5], jj[5], :]
            writer.writeFrame(full_frame)

        writer.close()
        subprocess.call(['ffmpeg', '-y', '-i', audio_file, '-i', out_file, '-map', '0:a', '-map', '1:v', '-c', 'copy',  '-shortest',
                                 out_file_sound_format.format(v_num)])
        os.remove(out_file)
        in_file = out_file_sound_format.format(v_num) 
        save_file = out_file_sound_360_format.format(v_num)
        metadata_utils.inject_metadata(in_file, save_file, metadata, console.append)
        os.remove(out_file_sound_format.format(v_num))
        os.rename(save_file, os.path.splitext(out_file)[0] + '.' + args.ext)

