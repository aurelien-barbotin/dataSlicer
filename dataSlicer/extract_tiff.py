#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:15:24 2019

@author: aurelien
"""

def get_stack_info(pars):
    npixels = pars['res_npixels']
    if pars['scan_type'] == 0:
        slow_npixels = pars['galvo_npixels']
        ext = (
            -pars['res_fov_um']/2,
            pars['res_fov_um']/2,
            pars['galvo_xmin_um'],
            pars['galvo_xmax_um']
            )
        labels = ('x', 'y')
        dim3_pos = pars['zpiezo_position']
        dim3_label = 'z'
        shape = (slow_npixels, npixels)
        psizes = (
            (ext[3] - ext[2])/shape[0],
            (ext[1] - ext[0])/shape[1],
        )
    elif pars['scan_type'] == 1:
        slow_npixels = pars['pz_npixels']
        ext = (
            -pars['res_fov_um']/2,
            pars['res_fov_um']/2,
            pars['pz_zmin_um'],
            pars['pz_zmax_um']
        )
        labels = ('x', 'z')
        dim3_pos = 'y'
        dim3_label = pars['galvo_position']
        shape = (slow_npixels, npixels)
        psizes = (
            (ext[3] - ext[2])/shape[0],
            (ext[1] - ext[0])/shape[1],
        )
    elif pars['scan_type'] == 2:
        slow_npixels = pars['galvo_npixels']
        pz_npixels = pars['pz_npixels']
        ext = (
            -pars['res_fov_um']/2,
            pars['res_fov_um']/2,
            pars['galvo_xmin_um'],
            pars['galvo_xmax_um'],
            pars['pz_zmin_um'],
            pars['pz_zmax_um']
            )
        labels = ('x', 'y', 'z')
        dim3_pos = None
        dim3_label = None
        shape = (pz_npixels, slow_npixels, npixels)
        psizes = (
            (ext[5] - ext[4])/shape[0],
            (ext[3] - ext[2])/shape[1],
            (ext[1] - ext[0])/shape[2],
            )
    else:
        raise ValueError(f"illegal scan_type: {pars['scan_type']}")

    return {
        'shape': shape,
        'ext': ext,
        'labels': labels,
        'pixel_sizes': psizes,
        'dim3_pos': dim3_pos,
        'dim3_label': dim3_label,
        }
    
import json
import numpy as np

from tifffile import TiffFile


class Scan:

    def __init__(self, file1):
        with TiffFile(file1) as t:
            data = t.asarray()
            tags = t.pages[0].tags
            d = {}
            for k, v in tags.items():
                d[k] = v.value
        self.data = data
        self.tags = d

        self.scanning_version = d['3712']
        self.scanning_commit = d['3713']
        self.scanning_date = d['3714']

        self.pars = json.loads(d['3715'])

        try:
            self.date = d['3716']
        except Exception:
            self.date = ''

        try:
            self.devs = json.loads(d['3717'])
        except Exception:
            self.devs = {}
        try:
            self.gui = json.loads(d['3718'])
        except Exception:
            self.gui = {}

        self.stack_info = get_stack_info(self.pars)

    def make_time_lapse(self):
        tinfo = self.time_info
        scan_time = tinfo['scan_time']
        seq_length = tinfo['seq_length']
        return np.linspace(0, scan_time*seq_length, seq_length)

    def make_meshgrid(self):
        shape = self.stack_info['shape']
        ext = self.stack_info['ext'][::-1]
        dds = []
        for i in range(len(ext)//2):
            dds.append(np.linspace(ext[2*i], ext[2*i + 1], shape[i]))
        self.dds = dds
        self.grid = np.meshgrid(*dds, indexing='ij')


if __name__ == '__main__':
    file = "/media/aurelien/362228e7-97c8-487c-b5d0-2d548f964446/Data/dSTED/2019_08_08_dendrites/ROI7/20190808_163054_STED_AOoff.tif"

    s = Scan(file)

    labels = s.stack_info['labels']
    shape = s.stack_info['shape']
    psizes = s.stack_info['pixel_sizes']
    ext = np.array(s.stack_info['ext'])

    j = 0
    N = len(labels)
    for i in range(N):
        print(
            f'{ext[j]} {ext[j + 1]} ' +
            f'({shape[N - i - 1]}; {psizes[N - i - 1]:.2f} nm) {labels[i]}')
        j += 2
    print()
