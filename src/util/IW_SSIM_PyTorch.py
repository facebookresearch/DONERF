# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

#BSD License
#
#Copyright (c) 2019, Xinyu Guo
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch.nn.functional as F
import pyrtools as pt
import numpy as np
import torch


class IW_SSIM():
    def __init__(self, iw_flag=True, Nsc=5, blSzX=3, blSzY=3, parent=True,
                 sigma_nsq=0.4, use_cuda=False, use_double=False):
        # MS-SSIM parameters
        self.K = [0.01, 0.03]
        self.L = 255
        self.weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        self.winsize = 11
        self.sigma = 1.5

        # IW-SSIM parameters
        self.iw_flag = iw_flag
        self.Nsc = Nsc    # scales
        self.blSzX = blSzX  # Neighbor size
        self.blSzY = blSzY
        self.parent = parent
        self.sigma_nsq = sigma_nsq

        self.bound = np.ceil((self.winsize-1)/2)
        self.bound1 = self.bound - np.floor((self.blSzX-1)/2)
        self.use_cuda = use_cuda
        self.use_double = use_double

        self.samplet = torch.tensor([1.0])
        if self.use_cuda:
            self.samplet = self.samplet.cuda()
        if self.use_double:
            self.samplet = self.samplet.double()
        self.samplen = np.array([1.0])
        if not self.use_double:
            self.samplen = self.samplen.astype('float32')

    def fspecial(self, fltr, ws, **kwargs):
        if fltr == 'uniform':
            return np.ones((ws, ws)) / ws**2

        elif fltr == 'gaussian':
            x, y = np.mgrid[-ws//2 + 1:ws//2 + 1, -ws//2 + 1:ws//2 + 1]
            g = np.exp(-((x**2 + y**2)/(2.0*kwargs['sigma']**2)))
            g[g < np.finfo(g.dtype).eps*g.max()] = 0
            assert g.shape == (ws, ws)
            den = g.sum()
            if den != 0:
                g /= den
            return g

        return None

    def get_pyrd(self, imgo, imgd):
        imgopr = {}
        imgdpr = {}
        lpo = pt.pyramids.LaplacianPyramid(imgo, height=5)
        lpd = pt.pyramids.LaplacianPyramid(imgd, height=5)
        for scale in range(1, self.Nsc + 1):
            imgopr[scale] = torch.from_numpy(lpo.pyr_coeffs[(scale-1, 0)]).unsqueeze(0).unsqueeze(0).type(self.samplet.type())
            imgdpr[scale] = torch.from_numpy(lpd.pyr_coeffs[(scale-1, 0)]).unsqueeze(0).unsqueeze(0).type(self.samplet.type())

        return imgopr, imgdpr

    def scale_qualty_maps(self, imgopr, imgdpr):

        ms_win = self.fspecial('gaussian', ws=self.winsize, sigma=self.sigma)
        ms_win = torch.from_numpy(ms_win).unsqueeze(0).unsqueeze(0).type(self.samplet.type())
        C1 = (self.K[0]*self.L)**2
        C2 = (self.K[1]*self.L)**2
        cs_map = {}
        for i in range(1, self.Nsc+1):
            imgo = imgopr[i]
            imgd = imgdpr[i]
            mu1 = F.conv2d(imgo, ms_win)
            mu2 = F.conv2d(imgd, ms_win)
            sigma12 = F.conv2d(imgo*imgd, ms_win) - mu1*mu2
            sigma1_sq = F.conv2d(imgo**2, ms_win) - mu1*mu1
            sigma2_sq = F.conv2d(imgd**2, ms_win) - mu2*mu2
            sigma1_sq = torch.max(torch.zeros(sigma1_sq.shape).type(self.samplet.type()), sigma1_sq)
            sigma2_sq = torch.max(torch.zeros(sigma2_sq.shape).type(self.samplet.type()), sigma2_sq)
            cs_map[i] = (2*sigma12+C2) / (sigma1_sq + sigma2_sq + C2)
            if i == self.Nsc:
                l_map = (2*mu1*mu2+C1) / (mu1**2+mu2**2+C1)

        return l_map, cs_map

    def roll(self, x, shift, dim):
        if dim == 0:
            return torch.cat((x[-shift:, :], x[:-shift, :]), dim)
        else:
            return torch.cat((x[:, -shift:], x[:, :-shift]), dim)

    def imenlarge2(self, im):
        _, _, M, N = im.shape
        # t1 = F.upsample(im, size=(int(4*M-3), int(4*N-3)), mode='bilinear')
        t1 = F.interpolate(im, size=(int(4*M-3), int(4*N-3)), mode='bilinear', align_corners=False)
        t2 = torch.zeros([1, 1, 4*M-1, 4*N-1]).type(self.samplet.type())
        t2[:, :, 1: -1, 1:-1] = t1
        t2[:, :, 0, :] = 2*t2[:, :, 1, :] - t2[:, :, 2, :]
        t2[:, :, -1, :] = 2*t2[:, :, -2, :] - t2[:, :, -3, :]
        t2[:, :, :, 0] = 2*t2[:, :, :, 1] - t2[:, :, :, 2]
        t2[:, :, :, -1] = 2*t2[:, :, :, -2] - t2[:, :, :, -3]
        imu = t2[:, :, ::2, ::2]

        return imu

    def info_content_weight_map(self, imgopr, imgdpr):

        tol = 1e-15
        iw_map = {}
        for scale in range(1, self.Nsc):

            imgo = imgopr[scale]
            imgd = imgdpr[scale]
            win = np.ones([self.blSzX, self.blSzY])
            win = win / np.sum(win)
            win = torch.from_numpy(win).unsqueeze(0).unsqueeze(0).type(self.samplet.type())
            padding = int((self.blSzX-1)/2)

            # Prepare for estimating IW-SSIM parameters
            mean_x = F.conv2d(imgo, win, padding=padding)
            mean_y = F.conv2d(imgd, win, padding=padding)
            cov_xy = F.conv2d(imgo*imgd, win, padding=padding) - mean_x*mean_y
            ss_x = F.conv2d(imgo**2, win, padding=padding) - mean_x**2
            ss_y = F.conv2d(imgd**2, win, padding=padding) - mean_y**2

            ss_x[ss_x < 0] = 0
            ss_y[ss_y < 0] = 0

            # Estimate gain factor and error
            g = cov_xy / (ss_x + tol)
            vv = (ss_y - g*cov_xy)
            g[ss_x < tol] = 0
            vv[ss_x < tol] = ss_y[ss_x < tol]
            ss_x[ss_x < tol] = 0
            g[ss_y < tol] = 0
            vv[ss_y < tol] = 0

            # Prepare parent band
            aux = imgo
            _, _, Nsy, Nsx = aux.shape
            prnt = (self.parent and scale < self.Nsc-1)
            BL = torch.zeros([1, 1, aux.shape[2], aux.shape[3], 1+prnt])
            if self.use_cuda:
                BL = BL.cuda()
            if self.use_double:
                BL = BL.double()

            BL[:, :, :, :, 0] = aux
            if prnt:
                auxp = imgopr[scale+1]
                auxp = self.imenlarge2(auxp)
                BL[:, :, :, :, 1] = auxp[:, :, 0:Nsy, 0:Nsx]
            imgo = BL
            _, _, nv, nh, nb = imgo.shape

            block = torch.tensor([win.shape[2], win.shape[3]])
            if self.use_cuda:
                block = block.cuda()

            # Group neighboring pixels
            nblv = nv-block[0]+1
            nblh = nh-block[1]+1
            nexp = nblv*nblh
            N = torch.prod(block) + prnt
            Ly = int((block[0]-1)//2)
            Lx = int((block[1]-1)//2)
            Y = torch.zeros([nexp, N]).type(self.samplet.type())

            n = -1
            for ny in range(-Ly, Ly+1):
                for nx in range(-Lx, Lx+1):
                    n = n + 1
                    temp = imgo[0, 0, :, :, 0]
                    foo1 = self.roll(temp, ny, 0)
                    foo = self.roll(foo1, nx, 1)
                    foo = foo[Ly: Ly+nblv, Lx: Lx+nblh]
                    Y[:, n] = foo.flatten()
            if prnt:
                n = n + 1
                temp = imgo[0, 0, :, :, 1]
                foo = temp
                foo = foo[Ly: Ly+nblv, Lx: Lx+nblh]
                Y[:, n] = foo.flatten()

            C_u = torch.mm(torch.transpose(Y, 0, 1), Y) / nexp.type(self.samplet.type())
            eig_values, H = torch.eig(C_u, eigenvectors=True)
            eig_values = eig_values.type(self.samplet.type())
            H = H.type(self.samplet.type())
            if self.use_double:
                L = torch.diag(eig_values[:, 0] * (eig_values[:, 0] > 0).double()) * torch.sum(eig_values) / ((torch.sum(eig_values[:,0] * (eig_values[:, 0] > 0).double())) + (torch.sum(eig_values[:, 0] * (eig_values[:, 0] > 0).double())==0))
            else:
                L = torch.diag(eig_values[:, 0] * (eig_values[:, 0] > 0).float()) * torch.sum(eig_values) / ((torch.sum(eig_values[:,0] * (eig_values[:, 0] > 0).float())) + (torch.sum(eig_values[:, 0] * (eig_values[:, 0] > 0).float())==0))
            C_u = torch.mm(torch.mm(H, L), torch.transpose(H, 0, 1))
            C_u_inv = torch.inverse(C_u)
            ss = (torch.mm(Y, C_u_inv))*Y / N.type(self.samplet.type())
            ss = torch.sum(ss, 1)
            ss = ss.view(nblv, nblh)
            ss = ss.unsqueeze(0).unsqueeze(0)
            g = g[:, :, Ly: Ly+nblv, Lx: Lx+nblh]
            vv = vv[:, :, Ly: Ly+nblv, Lx: Lx+nblh]

            # Calculate mutual information
            infow = torch.zeros(g.shape).type(self.samplet.type())
            for j in range(len(eig_values)):
                infow = infow + torch.log2(1 + ((vv + (1 + g*g)*self.sigma_nsq)*ss*eig_values[j, 0]+self.sigma_nsq*vv) / (self.sigma_nsq*self.sigma_nsq))
            infow[infow < tol] = 0
            iw_map[scale] = infow

        return iw_map

    def test(self, imgo, imgd):

        imgo = imgo.astype(self.samplen.dtype)
        imgd = imgd.astype(self.samplen.dtype)
        imgopr, imgdpr = self.get_pyrd(imgo, imgd)
        l_map, cs_map = self.scale_qualty_maps(imgopr, imgdpr)
        if self.iw_flag:
            iw_map = self.info_content_weight_map(imgopr, imgdpr)

        wmcs = []
        for s in range(1, self.Nsc+1):
            cs = cs_map[s]
            if s == self.Nsc:
                cs = cs_map[s]*l_map

            if self.iw_flag:
                if s < self.Nsc:
                    iw = iw_map[s]
                    if self.bound1 != 0:
                        iw = iw[:, :, int(self.bound1): -int(self.bound1), int(self.bound1): -int(self.bound1)]
                    else:
                        iw = iw[:, :, int(self.bound1):, int(self.bound1):]
                else:
                    iw = torch.ones(cs.shape).type(self.samplet.type())
                wmcs.append(torch.sum(cs*iw) / torch.sum(iw))
            else:
                wmcs.append(torch.mean(cs))

        wmcs = torch.tensor(wmcs).type(self.samplet.type())
        if not torch.is_tensor(self.weight):
            self.weight = torch.tensor(self.weight).type(self.samplet.type())
        score = torch.prod((torch.abs(wmcs))**(self.weight))

        return score
