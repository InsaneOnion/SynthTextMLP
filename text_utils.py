from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio
import os.path as osp
import random, os
import cv2
#import cPickle as cp
import _pickle as cp
import scipy.signal as ssig
import scipy.stats as sstat
import pygame, pygame.locals
from pygame import freetype
#import Image
from PIL import Image
import math
from common import *
import pickle
from trans_utils import My_Translator

def sample_weighted(p_dict):
    ps = list(p_dict.keys())
    return p_dict[np.random.choice(ps,p=ps)]

def move_bb(bbs, t):
    """
    Translate the bounding-boxes in by t_x,t_y.
    BB : 2x4xn
    T  : 2-long np.array
    """
    return bbs + t[:,None,None]

def crop_safe(arr, rect, bbs=[], pad=0):
    """
    ARR : arr to crop
    RECT: (x,y,w,h) : area to crop to
    BBS : nx4 xywh format bounding-boxes
    PAD : percentage to pad

    Does safe cropping. Returns the cropped rectangle and
    the adjusted bounding-boxes
    """
    rect = np.array(rect)
    rect[:2] -= pad
    rect[2:] += 2*pad
    v0 = [max(0,rect[0]), max(0,rect[1])]
    v1 = [min(arr.shape[0], rect[0]+rect[2]), min(arr.shape[1], rect[1]+rect[3])]
    arr = arr[v0[0]:v1[0],v0[1]:v1[1],...]
    if len(bbs) > 0:
        for i in range(len(bbs)):
            bbs[i,0] -= v0[0]
            bbs[i,1] -= v0[1]
        return arr, bbs
    else:
        return arr


class BaselineState(object):
    curve = lambda this, a: lambda x: a*x*x
    differential = lambda this, a: lambda x: 2*a*x
    a = [0.50, 0.05]

    def get_sample(self):
        """
        Returns the functions for the curve and differential for a and b
        """
        sgn = 1.0
        if np.random.rand() < 0.5:
            sgn = -1

        a = self.a[1]*np.random.randn() + sgn*self.a[0]
        return {
            'curve': self.curve(a),
            'diff': self.differential(a),
        }

class RenderFont(object):
    """
    Outputs a rasterized font sample.
        Output is a binary mask matrix cropped closesly with the font.
        Also, outputs ground-truth bounding boxes and text string
    """

    def __init__(self, data_dir='data'):
        # distribution over the type of text:
        # whether to get a single word, paragraph or a line:
        self.p_text = {0.0 : 'WORD',
                       0.0 : 'LINE',
                       1.0 : 'PARA'}

        ## TEXT PLACEMENT PARAMETERS:
        self.f_shrink = 0.90
        self.max_shrink_trials = 5 # 0.9^5 ~= 0.6
        # the minimum number of characters that should fit in a mask
        # to define the maximum font height.
        self.min_nchar = 2
        self.min_font_h = 16 #px : 0.6*12 ~ 7px <= actual minimum height
        self.max_font_h = 120 #px
        self.p_flat = 0.10

        # curved baseline:
        self.p_curved = 1.0
        self.baselinestate = BaselineState()

        # text-source : gets english text:
        self.text_source = TextSource(min_nchar=self.min_nchar,
                                      fn=osp.join(data_dir,'newsgroup/newsgroup.txt'))

        # get font-state object:
        self.font_state = FontState(data_dir)

        # initialize colorizer:
        from colorize3_poisson import Colorize
        self.colorizer = Colorize(data_dir)

        pygame.init()

    def render_bilingual_multiline(self, font, text_en, text_zh):
        """
        渲染配对的英文和中文文本为多行
        返回两个渲染结果：英文和中文
        """
        # 渲染英文文本
        surf_arr_en, words_en, bbs_en = self.render_multiline(font, text_en)
        
        # 渲染中文文本
        surf_arr_zh, words_zh, bbs_zh = self.render_multiline(font, text_zh)

        return surf_arr_en, words_en, bbs_en, surf_arr_zh, words_zh, bbs_zh

    def render_multiline(self,font,text):
        """
        renders multiline TEXT on the pygame surface SURF with the
        font style FONT.
        A new line in text is denoted by \n, no other characters are 
        escaped. Other forms of white-spaces should be converted to space.

        returns the updated surface, words and the character bounding boxes.
        """
        # get the number of lines
        lines = text.split('\n')
        lengths = [len(l) for l in lines]

        # font parameters:
        line_spacing = font.get_sized_height() + 1
        
        # initialize the surface to proper size:
        line_bounds = font.get_rect(lines[np.argmax(lengths)])
        fsize = (round(2.0*line_bounds.width), round(1.25*line_spacing*len(lines)))
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        bbs = []
        space = font.get_rect('O')
        x, y = 0, 0
        for l in lines:
            x = 0 # carriage-return
            y += line_spacing # line-feed

            for ch in l: # render each character
                if ch.isspace(): # just shift
                    x += space.width
                else:
                    # render the character
                    ch_bounds = font.render_to(surf, (x,y), ch)
                    ch_bounds.x = x + ch_bounds.x
                    ch_bounds.y = y - ch_bounds.y
                    x += ch_bounds.width
                    bbs.append(np.array(ch_bounds))

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # get the words:
        words = ' '.join(text.split())

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
        surf_arr = surf_arr.swapaxes(0,1)
        #self.visualize_bb(surf_arr,bbs)
        return surf_arr, words, bbs

    def render_bilingual_curved(self, font, word_text_en, word_text_zh):
        """
        使用弯曲基线渲染配对的英文和中文文本
        """
        # 移除多余的空格和换行符
        word_text_en = ' '.join(word_text_en.split())
        word_text_zh = ' '.join(word_text_zh.split())
        
        wl_en = len(word_text_en)
        wl_zh = len(word_text_zh)

        # 确保文本长度不超过10
        if (wl_en > 10 or wl_zh > 10 or wl_zh == 1 or 
            np.random.rand() > self.p_curved or 
            font.underline):  # 如果字体有下划线，不进行旋转
            # 分别渲染英文和中文文本
            en_result = self.render_multiline(font, word_text_en)
            zh_result = self.render_multiline(font, word_text_zh)
            
            # 确保两个渲染结果都成功
            if en_result is None or zh_result is None:
                return None
            
            # 解包渲染结果
            surf_arr_en, text_en, bb_en = en_result
            surf_arr_zh, text_zh, bb_zh = zh_result
            
            return surf_arr_en, text_en, bb_en, surf_arr_zh, text_zh, bb_zh

        # 创建两个独立的表面
        lspace = font.get_sized_height() + 1
        fsize_en = (round(2.0 * font.get_rect(word_text_en).width), round(3 * lspace))
        fsize_zh = (round(2.0 * font.get_rect(word_text_zh).width), round(3 * lspace))

        surf_en = pygame.Surface(fsize_en, pygame.locals.SRCALPHA, 32)
        surf_zh = pygame.Surface(fsize_zh, pygame.locals.SRCALPHA, 32)

        try:
            # 渲染英文文本
            mid_idx_en = wl_en // 2
            BS_en = self.baselinestate.get_sample()
            curve_en = [BS_en['curve'](i - mid_idx_en) for i in range(wl_en)]
            curve_en[mid_idx_en] = -np.sum(curve_en) / (wl_en - 1)
            rots_en = [-int(math.degrees(math.atan(BS_en['diff'](i - mid_idx_en) / (font.size / 2)))) for i in range(wl_en)]

            # 计算英文文本的总宽度和每个字符的宽度
            en_char_widths = [font.get_rect(ch).width for ch in word_text_en if not ch.isspace()]
            total_width_en = sum(en_char_widths)
            start_x_en = (surf_en.get_rect().width - total_width_en) // 2
            current_x_en = start_x_en

            bbs_en = []
            char_idx = 0
            for i, ch in enumerate(word_text_en):
                if ch.isspace():
                    current_x_en += font.get_rect(' ').width
                    continue
                
                rect = font.get_rect(ch)
                rect.x = current_x_en
                rect.centery = surf_en.get_rect().centery
                rect.centery += curve_en[i]

                try:
                    ch_bounds = font.render_to(surf_en, rect, ch, rotation=rots_en[i])
                except ValueError:
                    ch_bounds = font.render_to(surf_en, rect, ch)
                
                ch_bounds.x = rect.x + ch_bounds.x
                ch_bounds.y = rect.y - ch_bounds.y
                bbs_en.append(np.array(ch_bounds))
                
                current_x_en += en_char_widths[char_idx]
                char_idx += 1

            # 渲染中文文本
            mid_idx_zh = wl_zh // 2
            BS_zh = self.baselinestate.get_sample() 
            curve_zh = [BS_zh['curve'](i - mid_idx_zh) for i in range(wl_zh)]
            curve_zh[mid_idx_zh] = -np.sum(curve_zh) / (wl_zh - 1)
            rots_zh = [-int(math.degrees(math.atan(BS_zh['diff'](i - mid_idx_zh) / (font.size / 2)))) for i in range(wl_zh)]

            # 计算中文文本的总宽度和每个字符的宽度
            zh_char_widths = [font.get_rect(ch).width for ch in word_text_zh if not ch.isspace()]
            total_width_zh = sum(zh_char_widths)
            start_x_zh = (surf_zh.get_rect().width - total_width_zh) // 2
            current_x_zh = start_x_zh

            bbs_zh = []
            char_idx = 0
            for i, ch in enumerate(word_text_zh):
                if ch.isspace():
                    current_x_zh += font.get_rect(' ').width
                    continue
                
                rect = font.get_rect(ch)
                rect.x = current_x_zh
                rect.centery = surf_zh.get_rect().centery
                rect.centery += curve_zh[i]

                try:
                    ch_bounds = font.render_to(surf_zh, rect, ch, rotation=rots_zh[i])
                except ValueError:
                    ch_bounds = font.render_to(surf_zh, rect, ch)
                
                ch_bounds.x = rect.x + ch_bounds.x
                ch_bounds.y = rect.y - ch_bounds.y
                bbs_zh.append(np.array(ch_bounds))
                
                current_x_zh += zh_char_widths[char_idx]
                char_idx += 1

        except Exception as e:
            print(f"Error during rendering: {str(e)}")
            return None

        # 确保边界框数组不为空
        if not bbs_en or not bbs_zh:
            return None

        # 获取字符的联合矩形以进行裁剪
        bbs_en = np.array(bbs_en)
        bbs_zh = np.array(bbs_zh)
        
        r0_en = pygame.Rect(bbs_en[0])
        rect_union_en = r0_en.unionall(bbs_en)
        r0_zh = pygame.Rect(bbs_zh[0])
        rect_union_zh = r0_zh.unionall(bbs_zh)

        # 裁剪表面以适应文本
        surf_arr_en, bbs_en = crop_safe(pygame.surfarray.pixels_alpha(surf_en), rect_union_en, bbs_en, pad=5)
        surf_arr_zh, bbs_zh = crop_safe(pygame.surfarray.pixels_alpha(surf_zh), rect_union_zh, bbs_zh, pad=5)

        surf_arr_en = surf_arr_en.swapaxes(0, 1)
        surf_arr_zh = surf_arr_zh.swapaxes(0, 1)

        return surf_arr_en, word_text_en, bbs_en, surf_arr_zh, word_text_zh, bbs_zh

    def render_curved(self, font, word_text):
        """
        use curved baseline for rendering word
        """
        wl = len(word_text)
        isword = len(word_text.split())==1

        # do curved iff, the length of the word <= 10
        # print("word_text", word_text)
        # print("wl: ", wl)
        # print("isword: ", isword)
        if not isword or wl > 10 or wl == 1 or np.random.rand() > self.p_curved:
            return self.render_multiline(font, word_text)

        # create the surface:
        lspace = font.get_sized_height() + 1
        lbound = font.get_rect(word_text)
        fsize = (round(2.0*lbound.width), round(3*lspace))
        surf = pygame.Surface(fsize, pygame.locals.SRCALPHA, 32)

        # baseline state
        mid_idx = wl//2
        BS = self.baselinestate.get_sample()
        curve = [BS['curve'](i-mid_idx) for i in range(wl)]
        curve[mid_idx] = -np.sum(curve) / (wl-1)
        rots  = [-int(math.degrees(math.atan(BS['diff'](i-mid_idx)/(font.size/2)))) for i in range(wl)]

        bbs = []
        # place middle char
        rect = font.get_rect(word_text[mid_idx])
        rect.centerx = surf.get_rect().centerx
        rect.centery = surf.get_rect().centery + rect.height
        # print("curve[mid_idx]: ", curve[mid_idx])
        rect.centery +=  curve[mid_idx]
        ch_bounds = font.render_to(surf, rect, word_text[mid_idx], rotation=rots[mid_idx])
        ch_bounds.x = rect.x + ch_bounds.x
        ch_bounds.y = rect.y - ch_bounds.y
        mid_ch_bb = np.array(ch_bounds)

        # render chars to the left and right:
        last_rect = rect
        ch_idx = []
        for i in range(wl):
            #skip the middle character
            if i==mid_idx: 
                bbs.append(mid_ch_bb)
                ch_idx.append(i)
                continue

            if i < mid_idx: #left-chars
                i = mid_idx-1-i
            elif i==mid_idx+1: #right-chars begin
                last_rect = rect

            ch_idx.append(i)
            ch = word_text[i]

            newrect = font.get_rect(ch)
            newrect.y = last_rect.y
            if i > mid_idx:
                newrect.topleft = (last_rect.topright[0]+2, newrect.topleft[1])
            else:
                newrect.topright = (last_rect.topleft[0]-2, newrect.topleft[1])
            newrect.centery = max(newrect.height, min(fsize[1] - newrect.height, newrect.centery + curve[i]))
            try:
                bbrect = font.render_to(surf, newrect, ch, rotation=rots[i])
            except ValueError:
                bbrect = font.render_to(surf, newrect, ch)
            bbrect.x = newrect.x + bbrect.x
            bbrect.y = newrect.y - bbrect.y
            bbs.append(np.array(bbrect))
            last_rect = newrect
        
        # correct the bounding-box order:
        bbs_sequence_order = [None for i in ch_idx]
        for idx,i in enumerate(ch_idx):
            bbs_sequence_order[i] = bbs[idx]
        bbs = bbs_sequence_order

        # get the union of characters for cropping:
        r0 = pygame.Rect(bbs[0])
        rect_union = r0.unionall(bbs)

        # crop the surface to fit the text:
        bbs = np.array(bbs)
        surf_arr, bbs = crop_safe(pygame.surfarray.pixels_alpha(surf), rect_union, bbs, pad=5)
        surf_arr = surf_arr.swapaxes(0,1)
        return surf_arr, word_text, bbs


    def get_nline_nchar(self,mask_size,font_height,font_width):
        """
        Returns the maximum number of lines and characters which can fit
        in the MASK_SIZED image.
        """
        H,W = mask_size
        nline = int(np.ceil(H/(2*font_height)))
        nchar = int(np.floor(W/font_width))
        return nline,nchar

    def place_text(self, text_arrs, back_arr, bbs):
        areas = [-np.prod(ta.shape) for ta in text_arrs]
        order = np.argsort(areas)

        locs = [None for i in range(len(text_arrs))]
        out_arr = np.zeros_like(back_arr)
        for i in order:            
            ba = np.clip(back_arr.copy().astype(np.float), 0, 255)
            ta = np.clip(text_arrs[i].copy().astype(np.float), 0, 255)
            ba[ba > 127] = 1e8
            intersect = ssig.fftconvolve(ba,ta[::-1,::-1],mode='valid')
            safemask = intersect < 1e8

            if not np.any(safemask): # no collision-free position:
                #warn("COLLISION!!!")
                return back_arr,locs[:i],bbs[:i],order[:i]

            minloc = np.transpose(np.nonzero(safemask))
            loc = minloc[np.random.choice(minloc.shape[0]),:]
            locs[i] = loc

            # update the bounding-boxes:
            bbs[i] = move_bb(bbs[i],loc[::-1])

            # blit the text onto the canvas
            w,h = text_arrs[i].shape
            out_arr[loc[0]:loc[0]+w,loc[1]:loc[1]+h] += text_arrs[i]

        return out_arr, locs, bbs, order

    def robust_HW(self,mask):
        m = mask.copy()
        m = (~mask).astype('float')/255
        rH = np.median(np.sum(m,axis=0))
        rW = np.median(np.sum(m,axis=1))
        return rH,rW

    def sample_font_height_px(self,h_min,h_max):
        if np.random.rand() < self.p_flat:
            rnd = np.random.rand()
        else:
            rnd = np.random.beta(2.0,2.0)

        h_range = h_max - h_min
        f_h = np.floor(h_min + h_range*rnd)
        return f_h

    def bb_xywh2coords(self,bbs):
        """
        Takes an nx4 bounding-box matrix specified in x,y,w,h
        format and outputs a 2x4xn bb-matrix, (4 vertices per bb).
        """
        n,_ = bbs.shape
        coords = np.zeros((2,4,n))
        for i in range(n):
            coords[:,:,i] = bbs[i,:2][:,None]
            coords[0,1,i] += bbs[i,2]
            coords[:,2,i] += bbs[i,2:4]
            coords[1,3,i] += bbs[i,3]
        return coords

    def get_min_h(self, bb, text):
        """
        获取文本的最小高度
        bb: 边界框坐标 (2x4xn)
        text: 文本字符串
        """
        # 移除换行符和多余空格
        text = ' '.join(text.split())
        
        # 确保边界框是正确的格式
        if isinstance(bb, list):
            bb = np.array(bb)
        
        # 如果是 Nx4 格式，转换为 2x4xN
        if len(bb.shape) == 2 and bb.shape[1] == 4:
            bb = self.bb_xywh2coords(bb)
            
        # 计算每个字符的高度
        heights = np.linalg.norm(bb[:,3,:] - bb[:,0,:], axis=0)
        
        # 只考虑字母数字字符的高度
        alnum = np.array([ch.isalnum() for ch in text])
        
        # 确保 alnum 和 heights 长度匹配
        if len(alnum) > len(heights):
            # 如果文本长度大于边界框数量，截断文本
            alnum = alnum[:len(heights)]
        elif len(alnum) < len(heights):
            # 如果边界框数量大于文本长度，截断边界框
            heights = heights[:len(alnum)]
        
        if np.any(alnum):  # 如果有字母数字字符
            heights = heights[alnum]
            return np.min(heights)
        else:  # 如果没有字母数字字符
            return np.min(heights) if heights.size > 0 else 0

    def render_bilingual_sample(self, font, mask):
        """
        Places text in the "collision-free" region as indicated
        in the mask -- 255 for unsafe, 0 for safe.
        """
        #H,W = mask.shape
        H,W = self.robust_HW(mask)
        f_asp = self.font_state.get_aspect_ratio(font)
        print("f_asp: ", f_asp)

        # find the maximum height in pixels:
        max_font_h = min(0.9*H, (1/f_asp)*W/(self.min_nchar+1))
        max_font_h = min(max_font_h, self.max_font_h)
        if max_font_h < self.min_font_h: # not possible to place any text here
            return #None

        # let's just place one text-instance for now
        ## TODO : change this to allow multiple text instances?
        i = 0
        while i < self.max_shrink_trials and max_font_h > self.min_font_h:
            # if i > 0:
            #     print colorize(Color.BLUE, "shrinkage trial : %d"%i, True)

            # sample a random font-height:
            f_h_px = self.sample_font_height_px(self.min_font_h, max_font_h)
            #print "font-height : %.2f (min: %.2f, max: %.2f)"%(f_h_px, self.min_font_h,max_font_h)
            # convert from pixel-height to font-point-size:
            f_h = self.font_state.get_font_size(font, f_h_px)

            # update for the loop
            max_font_h = f_h_px 
            i += 1

            font.size = f_h # set the font-size

            # compute the max-number of lines/chars-per-line:
            nline,nchar = self.get_nline_nchar(mask.shape[:2],f_h,f_h*f_asp)
            #print "  > nline = %d, nchar = %d"%(nline, nchar)

            assert nline >= 1 and nchar >= self.min_nchar

            # sample text:
            text_type = sample_weighted(self.p_text)
            src_text = self.text_source.sample(nline,nchar,text_type)
            if len(src_text)==0 or np.any([len(line)==0 for line in src_text]):
                continue
            #print colorize(Color.GREEN, text)

            # translate the text to another language
            print("src_text: ", src_text)
            self.translate = My_Translator("en", "zh-cn")
            handled_text, char_ratio, centered = self.translate.handle_src_text(src_text)
            print("handled_text: ", handled_text)
            print("char_ratio: ", char_ratio)
            print("centered: ", centered)
            tgt_text = self.translate.translate(handled_text)
            print("translated_tgt_text: ", tgt_text)
            tgt_text = self.translate.handle_tgt_text(tgt_text, char_ratio, centered)
            print("handled_tgt_text: ", tgt_text)

            # render the text:
            render_result = self.render_bilingual_curved(font, src_text, tgt_text)
            if render_result is None:  # 渲染失败
                continue
            
            src_txt_arr, src_text, src_bb, tgt_txt_arr, tgt_text, tgt_bb = render_result
            
            # 转换边界框坐标格式
            src_bb = self.bb_xywh2coords(src_bb)
            tgt_bb = self.bb_xywh2coords(tgt_bb)

            # make sure that the text-array is not bigger than mask array:
            if (np.any(np.r_[src_txt_arr.shape[:2]] > np.r_[mask.shape[:2]]) or 
                np.any(np.r_[tgt_txt_arr.shape[:2]] > np.r_[mask.shape[:2]])):
                continue

            # 将 ndarray 转为 PIL 图像
            src_img = Image.fromarray(src_txt_arr)
            tgt_img = Image.fromarray(tgt_txt_arr)

            # 保存为 PNG 图片
            src_img.save('src_text_mask_pil.png')
            tgt_img.save('tgt_text_mask_pil.png')
            
            # 首先尝试放置英文文本
            src_text_mask, src_loc, src_bb, _ = self.place_text([src_txt_arr], mask, [src_bb])
            
            if len(src_loc) > 0:  # 如果英文文本放置成功
                # 计算中文文本的位置，使其点与英文文本对齐
                src_center = src_loc[0] + np.array([src_txt_arr.shape[0]//2, src_txt_arr.shape[1]//2])
                tgt_offset = src_center - np.array([tgt_txt_arr.shape[0]//2, tgt_txt_arr.shape[1]//2])
                
                # 创建中文文本的mask
                tgt_text_mask = np.zeros_like(mask)
                h, w = tgt_txt_arr.shape
                y, x = tgt_offset
                
                # 确保位置在有效范围内
                if (y >= 0 and x >= 0 and 
                    y + h <= mask.shape[0] and 
                    x + w <= mask.shape[1]):
                    # 放置中文文本
                    tgt_text_mask[y:y+h, x:x+w] = tgt_txt_arr
                    
                    # 调整中文文本的边界框位置
                    if isinstance(tgt_bb, list):
                        tgt_bb = np.array(tgt_bb)
                    if len(tgt_bb.shape) == 2 and tgt_bb.shape[1] == 4:
                        tgt_bb = self.bb_xywh2coords(tgt_bb)
                    
                    # 调整边界框位置
                    tgt_bb = tgt_bb.copy()
                    for i in range(tgt_bb.shape[-1]):
                        tgt_bb[:,:,i] += tgt_offset[::-1][:,None]  # 注意坐标转换
                    
                    # 确保边界框是 (2,4,n) 格式
                    if isinstance(src_bb, list):
                        src_bb = np.array(src_bb[0])  # 因为place_text返回的是列表
                    if len(src_bb.shape) == 2 and src_bb.shape[1] == 4:
                        src_bb = self.bb_xywh2coords(src_bb)
                    
                    # 最终检查边界框格式
                    if (len(src_bb.shape) == 3 and src_bb.shape[0] == 2 and src_bb.shape[1] == 4 and
                        len(tgt_bb.shape) == 3 and tgt_bb.shape[0] == 2 and tgt_bb.shape[1] == 4):
                        
                        # 获取文本的最小高度，用于确定渲染风格
                        src_min_h = self.get_min_h(src_bb, src_text)
                        tgt_min_h = self.get_min_h(tgt_bb, tgt_text)
                        
                        try:
                            # 直接返回掩码，让 place_text 处理颜色渲染
                            return {
                                "src": [src_text_mask, src_loc[0], src_bb, src_text],  # 返回原始掩码
                                "tgt": [tgt_text_mask, tgt_offset, tgt_bb, tgt_text]   # 返回原始掩码
                            }
                        except Exception as e:
                            print(f"Error during colorization: {str(e)}")
                            return None
        
        return None


    def visualize_bb(self, text_arr, bbs):
        ta = text_arr.copy()
        for r in bbs:
            cv2.rectangle(ta, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), color=128, thickness=1)
        plt.imshow(ta,cmap='gray')
        plt.show()


class FontState(object):
    """
    Defines the random state of the font rendering  
    """
    size = [50, 10]  # normal dist mean, std
    underline = 0.05
    strong = 0.5
    oblique = 0.2
    wide = 0.5
    strength = [0.05, 0.1]  # uniform dist in this interval
    underline_adjustment = [1.0, 2.0]  # normal dist mean, std
    kerning = [2, 5, 0, 20]  # beta distribution alpha, beta, offset, range (mean is a/(a+b))
    border = 0.25
    random_caps = -1 ## don't recapitalize : retain the capitalization of the lexicon
    capsmode = [str.lower, str.upper, str.capitalize]  # lower case, upper case, proper noun
    curved = 0.2
    random_kerning = 0.2
    random_kerning_amount = 0.1

    def __init__(self, data_dir='data'):

        char_freq_path = osp.join(data_dir, 'models/char_freq.cp')        
        font_model_path = osp.join(data_dir, 'models/font_px2pt.cp')

        # get character-frequencies in the English language:
        with open(char_freq_path,'rb') as f:
            self.char_freq = cp.load(f)
            # u = pickle._Unpickler(f)
            # u.encoding = 'latin1'
            # p = u.load()
            # self.char_freq = p

        # get the model to convert from pixel to font pt size:
        with open(font_model_path,'rb') as f:
            self.font_model = cp.load(f)
            # u = pickle._Unpickler(f)
            # u.encoding = 'latin1'
            # p = u.load()
            # self.font_model = p
            
        # get the names of fonts to use:
        self.FONT_LIST = osp.join(data_dir, 'fonts/fontlist.txt')
        self.fonts = [os.path.join(data_dir,'fonts',f.strip()) for f in open(self.FONT_LIST)]


    def get_aspect_ratio(self, font, size=None):
        """
        Returns the median aspect ratio of each character of the font.
        """
        if size is None:
            size = 12 # doesn't matter as we take the RATIO
        chars = ''.join(self.char_freq.keys())
        w = np.array(self.char_freq.values())

        # get the [height,width] of each character:
        try:
            sizes = font.get_metrics(chars,size)
            good_idx = [i for i in range(len(sizes)) if sizes[i] is not None]
            sizes,w = [sizes[i] for i in good_idx], w[good_idx]
            sizes = np.array(sizes).astype('float')[:,[3,4]]        
            r = np.abs(sizes[:,1]/sizes[:,0]) # width/height
            good = np.isfinite(r)
            r = r[good]
            w = w[good]
            w /= np.sum(w)
            r_avg = np.sum(w*r)
            return r_avg
        except:
            return 1.0

    def get_font_size(self, font, font_size_px):
        """
        Returns the font-size which corresponds to FONT_SIZE_PX pixels font height.
        """
        m = self.font_model[font.name]
        return m[0]*font_size_px + m[1] #linear model


    def sample(self):
        """
        Samples from the font state distribution
        """
        return {
            'font': self.fonts[int(np.random.randint(0, len(self.fonts)))],
            'size': self.size[1]*np.random.randn() + self.size[0],
            'underline': np.random.rand() < self.underline,
            'underline_adjustment': max(2.0, min(-2.0, self.underline_adjustment[1]*np.random.randn() + self.underline_adjustment[0])),
            'strong': np.random.rand() < self.strong,
            'oblique': np.random.rand() < self.oblique,
            'strength': (self.strength[1] - self.strength[0])*np.random.rand() + self.strength[0],
            'char_spacing': int(self.kerning[3]*(np.random.beta(self.kerning[0], self.kerning[1])) + self.kerning[2]),
            'border': np.random.rand() < self.border,
            'random_caps': np.random.rand() < self.random_caps,
            'capsmode': random.choice(self.capsmode),
            'curved': np.random.rand() < self.curved,
            'random_kerning': np.random.rand() < self.random_kerning,
            'random_kerning_amount': self.random_kerning_amount,
        }

    def init_font(self,fs):
        """
        Initializes a pygame font.
        FS : font-state sample
        """
        font = freetype.Font(fs['font'], size=fs['size'])
        font.underline = fs['underline']
        font.underline_adjustment = fs['underline_adjustment']
        font.strong = fs['strong']
        font.oblique = fs['oblique']
        font.strength = fs['strength']
        char_spacing = fs['char_spacing']
        font.antialiased = True
        font.origin = True
        return font


class TextSource(object):
    """
    Provides text for words, paragraphs, sentences.
    """
    def __init__(self, min_nchar, fn):
        """
        TXT_FN : path to file containing text data.
        """
        self.min_nchar = min_nchar
        self.fdict = {'WORD':self.sample_word,
                      'LINE':self.sample_line,
                      'PARA':self.sample_para}

        with open(fn,'r') as f:
            self.txt = [l.strip() for l in f.readlines()]

        # distribution over line/words for LINE/PARA:
        self.p_line_nline = np.array([0.85, 0.10, 0.05])
        self.p_line_nword = [4,3,12]  # normal: (mu, std)
        self.p_para_nline = [1.0,1.0] # [1.7,3.0] # beta: (a, b), max_nline
        self.p_para_nword = [1.7,3.0,10] # beta: (a,b), max_nword

        # probability to center-align a paragraph:
        self.center_para = 0.5


    def check_symb_frac(self, txt, f=0.35):
        """
        T/F return : T iff fraction of symbol/special-charcters in
                     txt is less than or equal to f (default=0.25).
        """
        return np.sum([not ch.isalnum() for ch in txt])/(len(txt)+0.0) <= f

    def is_good(self, txt, f=0.35):
        """
        T/F return : T iff the lines in txt (a list of txt lines)
                     are "valid".
                     A given line l is valid iff:
                         1. It is not empty.
                         2. symbol_fraction > f
                         3. Has at-least self.min_nchar characters
                         4. Not all characters are i,x,0,O,-
        """
        def is_txt(l):
            """
            judgment if the txt all composite with meaningless char
            """
            char_ex = ['i','I','o','O','0','-']
            chs = [ch in char_ex for ch in l]
            return not np.all(chs)

        return [ (len(l)> self.min_nchar
                 and self.check_symb_frac(l,f)
                 and is_txt(l)) for l in txt ]

    def center_align(self, lines):
        """
        PADS lines with space to center align them
        lines : list of text-lines.
        """
        ls = [len(l) for l in lines]
        max_l = max(ls)
        for i in range(len(lines)):
            l = lines[i].strip()
            dl = max_l-ls[i]
            lspace = dl//2
            rspace = dl-lspace
            lines[i] = ' '*lspace+l+' '*rspace
        return lines

    def get_lines(self, nline, nword, nchar_max, f=0.35, niter=100):
        def h_lines(niter=100):
            lines = ['']
            iter = 0
            while not np.all(self.is_good(lines,f)) and iter < niter:
                iter += 1
                line_start = np.random.choice(len(self.txt)-nline)
                lines = [self.txt[line_start+i] for i in range(nline)]
            return lines

        lines = ['']
        iter = 0
        while not np.all(self.is_good(lines,f)) and iter < niter:
            iter += 1
            # random choice n lines that satisfied the filter strategy
            lines = h_lines(niter=100)
            # get words per line:
            nline = len(lines)
            for i in range(nline):
                words = lines[i].split()
                dw = len(words)-nword[i]
                # if the words in the line overflow
                if dw > 0:
                    # randomly chop-off dw words from front
                    first_word_index = random.choice(range(dw+1))
                    lines[i] = ' '.join(words[first_word_index:first_word_index+nword[i]])

                while len(lines[i]) > nchar_max: #chop-off characters from end:
                    if not np.any([ch.isspace() for ch in lines[i]]):
                        lines[i] = ''
                    else:
                        lines[i] = lines[i][:len(lines[i])-lines[i][::-1].find(' ')].strip()
        
        if not np.all(self.is_good(lines,f)):
            return #None
        else:
            return lines

    def sample(self, nline_max,nchar_max,kind='WORD'):
        return self.fdict[kind](nline_max,nchar_max)
        
    def sample_word(self,nline_max,nchar_max,niter=100):
        rand_line = self.txt[np.random.choice(len(self.txt))]                
        words = rand_line.split()
        rand_word = random.choice(words)

        iter = 0
        while iter < niter and (not self.is_good([rand_word])[0] or len(rand_word)>nchar_max):
            rand_line = self.txt[np.random.choice(len(self.txt))]                
            words = rand_line.split()
            rand_word = random.choice(words)
            iter += 1

        if not self.is_good([rand_word])[0] or len(rand_word)>nchar_max:
            return []
        else:
            return rand_word


    def sample_line(self,nline_max,nchar_max):
        nline = nline_max+1
        while nline > nline_max:
            nline = np.random.choice([1,2,3], p=self.p_line_nline)

        # get number of words:
        nword = [self.p_line_nword[2]*sstat.beta.rvs(a=self.p_line_nword[0], b=self.p_line_nword[1])
                 for _ in range(nline)]
        nword = [max(1,int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            return '\n'.join(lines)
        else:
            return []

    def sample_para(self,nline_max,nchar_max):
        """
        nline_max, nchar_max: the max-number of lines/chars-per-line                
        """
        # sample from beta distribution to determine the no. of lines randomly
        nline = nline_max*sstat.beta.rvs(a=self.p_para_nline[0], b=self.p_para_nline[1])
        nline = max(1, int(np.ceil(nline)))

        # sample from beta distribution to determine the no. of words per line in a paragraph randomly
        nword = [self.p_para_nword[2]*sstat.beta.rvs(a=self.p_para_nword[0], b=self.p_para_nword[1])
                 for _ in range(nline)]
        nword = [max(1,int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            # center align the paragraph-text:
            if np.random.rand() < self.center_para:
                lines = self.center_align(lines)
            return '\n'.join(lines)
        else:
            return []

