# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import re
import imgui
import numpy as np
import threading

from PIL import Image
from . import renderer
from .utils import EasyDict
from .gui_utils import imgui_utils

import slideflow as sf

#----------------------------------------------------------------------------

def _locate_results(pattern):
    return pattern

#----------------------------------------------------------------------------

class SlideWidget:
    def __init__(self, viz):
        self.viz                    = viz
        self.search_dirs            = []
        self.cur_slide              = None
        self.user_slide             = ''
        self.project_slides         = []
        self.browse_cache           = dict()
        self.browse_refocus         = False
        self.normalize_wsi          = False
        self.norm_idx               = 0
        self.qc_idx                 = 0
        self.qc_mask                = None
        self.alpha                  = 1.0
        self.stride                 = 1
        self.show_slide_filter      = False
        self.show_tile_filter       = False
        self.gs_fraction            = sf.slide.DEFAULT_GRAYSPACE_FRACTION
        self.gs_threshold           = sf.slide.DEFAULT_GRAYSPACE_THRESHOLD
        self.ws_fraction            = sf.slide.DEFAULT_WHITESPACE_FRACTION
        self.ws_threshold           = sf.slide.DEFAULT_WHITESPACE_THRESHOLD
        self.num_total_rois         = 0
        self._filter_grid           = None
        self._filter_thread         = None
        self._capturing_ws_thresh   = None
        self._capturing_gs_thresh   = None
        self._capturing_stride      = None
        self._use_rois              = True
        self._rendering_message = "Calculating tile filter..."
        self._all_normalizer_methods = [
            'reinhard',
            'reinhard_fast',
            'reinhard_mask',
            'reinhard_fast_mask',
            'macenko',
            'vahadane_spams',
            'vahadane_sklearn',
            'augment']
        self._all_normalizer_methods_str = [
            'Reinhard',
            'Reinhard (Fast)',
            'Reinhard Mask',
            'Reinhard Mask (Fast)',
            'Macenko',
            'Vahadane (SPAMS)',
            'Vahadane (Sklearn)',
            'Augment']
        self._normalizer_methods = self._all_normalizer_methods
        self._normalizer_methods_str = self._all_normalizer_methods_str
        self._qc_methods_str    = ['Blur filter', "Otsu threshold", "Blur + Otsu"]
        self._qc_methods        = ['blur', 'otsu', 'both']
        self.load('', ignore_errors=True)

    @property
    def show_overlay(self):
        return self.show_slide_filter or self.show_tile_filter

    @property
    def _thread_is_running(self):
        return self._filter_thread is not None and self._filter_thread.is_alive()

    def load(self, slide, ignore_errors=False):
        viz = self.viz
        self.reset_tile_filter_and_join_thread()
        viz.clear_result()
        viz.skip_frame() # The input field will change on next frame.
        if slide == '':
            viz.result = EasyDict(message='No slide loaded')
            return
        try:
            name = slide.replace('\\', '/').split('/')[-1]
            self.cur_slide = slide
            self.user_slide = slide
            self._use_rois = True
            self.viz.set_message(f'Loading {name}...')
            viz.defer_rendering()
            viz._reload_wsi(slide, stride=self.stride, use_rois=self._use_rois)
            viz.reset_thumb()
            viz.heatmap_widget.reset()
            self.num_total_rois = len(viz.wsi.rois)

            # Generate WSI thumbnail.
            hw_ratio = (viz.wsi.dimensions[0] / viz.wsi.dimensions[1])
            max_width = int(min(800 - viz.spacing*2, (800 - viz.spacing*2) / hw_ratio))
            viz.wsi_thumb = np.asarray(viz.wsi.thumb(width=max_width))
            self.viz.clear_message(f'Loading {name}...')

        except Exception:
            self.cur_slide = None
            self.user_slide = slide
            self.viz.clear_message(f'Loading {name}...')
            viz.result = EasyDict(error=renderer.CapturedException())
            if not ignore_errors:
                raise

    def _clear_images(self):
        self._overlay_tex_img    = None

    def _join_filter_thread(self):
        if self._filter_thread is not None:
            self._filter_thread.join()
        self._filter_thread = None

    def _start_filter_thread(self):
        self._join_filter_thread()
        self._filter_thread = threading.Thread(target=self._build_filter_grid)
        self._filter_thread.start()

    def refresh_stride(self):
        self.reset_tile_filter_and_join_thread()
        self.viz.clear_overlay()
        self.show_tile_filter = False
        self.show_slide_filter = False
        self.viz._reload_wsi(stride=self.stride, use_rois=self._use_rois)

    def hide_model_normalizer(self):
        self._normalizer_methods = self._all_normalizer_methods
        self._normalizer_methods_str = self._all_normalizer_methods_str
        self.norm_idx = min(self.norm_idx, len(self._normalizer_methods)-1)

    def show_model_normalizer(self):
        self._normalizer_methods = self._all_normalizer_methods + ['model']
        self._normalizer_methods_str = self._all_normalizer_methods_str + ['<Model>']

    def change_normalizer(self):
        method = self._normalizer_methods[self.norm_idx]
        if method == 'model':
            self.viz._normalizer = sf.util.get_model_normalizer(self.viz._model_path)
        else:
            self.viz._normalizer = sf.norm.autoselect(method, source='v2')

    def render_to_overlay(self, mask, correct_wsi_dim=None):
        """Renders boolean mask as an overlay, where:

            True = show tile from slide
            False = show black box
        """
        assert mask.dtype == bool
        alpha = (~mask).astype(np.uint8) * 255
        black = np.zeros(list(mask.shape) + [3], dtype=np.uint8)
        self.viz.overlay_heatmap = np.dstack((black, alpha))
        if correct_wsi_dim:
            full_extract = self.viz.wsi.tile_um / self.viz.wsi.mpp
            wsi_factor = full_extract / self.viz.wsi.stride_div
            self.viz._overlay_wsi_dim = (int(full_extract + wsi_factor * (mask.shape[1]-1)),
                                         int(full_extract + wsi_factor * (mask.shape[0]-1)))
        else:
            self.viz._overlay_wsi_dim = None

    def reset_tile_filter_and_join_thread(self):
        self._join_filter_thread()
        self._clear_images()
        self._filter_grid = None
        self._filter_thread = None
        self._ws_grid = None
        self._gs_grid = None

    def _build_filter_grid(self):
        if self.viz.wsi is not None:
            self.viz.set_message(self._rendering_message)
            generator = self.viz.wsi.build_generator(
                img_format='numpy',
                grayspace_fraction=sf.slide.FORCE_CALCULATE_GRAYSPACE,
                grayspace_threshold=self.gs_threshold,
                whitespace_fraction=sf.slide.FORCE_CALCULATE_WHITESPACE,
                whitespace_threshold=self.ws_threshold,
                shuffle=False,
                dry_run=True)
            if not generator:
                self.viz.clear_message(self._rendering_message)
                return
            # Returns boolean grid, where:
            #   True = tile will be extracted
            #   False = tile will be discarded (failed QC)
            self._filter_grid = np.transpose(self.viz.wsi.grid).astype(np.bool)
            self._ws_grid = np.zeros_like(self._filter_grid, dtype=np.float)
            self._gs_grid = np.zeros_like(self._filter_grid, dtype=np.float)
            self.render_to_overlay(self._filter_grid, correct_wsi_dim=True)
            for tile in generator():
                x = tile['grid'][0]
                y = tile['grid'][1]
                gs = tile['gs_fraction']
                ws = tile['ws_fraction']
                try:
                    self._ws_grid[y][x] = ws
                    self._gs_grid[y][x] = gs
                    if gs > self.gs_fraction or ws > self.ws_fraction:
                        self._filter_grid[y][x] = False
                        self.render_to_overlay(self._filter_grid, correct_wsi_dim=True)
                except TypeError:
                    # Occurs when the _ws_grid is reset, e.g. the slide was re-loaded.
                    print("Aborting tile filter calculation")
                    self.viz.clear_message(self._rendering_message)
                    return
            self.viz.clear_message(self._rendering_message)

    def _render_slide_filter(self):
        """Render the slide filter (QC) to screen."""
        self.viz.heatmap_widget.show = False
        self._clear_images()
        self.viz._overlay_wsi_dim = None
        self.render_to_overlay(self.qc_mask)

    def _render_tile_filter(self):
        self.viz.heatmap_widget.show = False
        self._join_filter_thread()
        self.render_to_overlay(self._filter_grid, correct_wsi_dim=True)

    def update_tile_filter(self):
        if self.show_tile_filter:
            self.viz.heatmap_widget.show = False
            self._join_filter_thread()
            self._clear_images()
            if not self.show_slide_filter:
                self.viz.overlay_heatmap = None
            if self._filter_grid is None and self.viz.wsi is not None:
                self._start_filter_thread()
            elif self._filter_grid is not None:
                self._render_tile_filter()
        else:
            self._clear_images()
            if self.show_slide_filter:
                self._render_slide_filter()

    def update_slide_filter(self):
        self.viz.wsi.remove_qc()
        self._join_filter_thread()

        # Update the slide QC
        if self.show_slide_filter and self.viz.wsi is not None:
            self.viz.heatmap_widget.show = False
            self.qc_mask = ~np.asarray(self.viz.wsi.qc(self._qc_methods[self.qc_idx]), dtype=np.bool)
        else:
            self.qc_mask = None

        # Update the tile filter since the QC method has changed
        self.reset_tile_filter_and_join_thread()
        if self.show_tile_filter:
            self.update_tile_filter()

        # Render the slide filter
        if self.show_slide_filter and not self.show_tile_filter:
            self._render_slide_filter()

    def refresh_gs_ws(self):
        self._join_filter_thread()
        if self._ws_grid is not None:
            # Returns boolean grid, where:
            #   True = tile will be extracted
            #   False = tile will be discarded (failed QC)
            self._filter_grid = np.transpose(self.viz.wsi.grid).astype(np.bool)
            for y in range(self._ws_grid.shape[0]):
                for x in range(self._ws_grid.shape[1]):
                    ws = self._ws_grid[y][x]
                    gs = self._gs_grid[y][x]
                    if gs > self.gs_fraction or ws > self.ws_fraction:
                        self._filter_grid[y][x] = False
            if self.show_tile_filter:
                self.render_to_overlay(self._filter_grid, correct_wsi_dim=True)

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            imgui.text('Slide')
            imgui.same_line(viz.label_w)
            changed, self.user_slide = imgui_utils.input_text('##slide', self.user_slide, 1024,
                flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                width=(-1 - viz.button_w * 1 - viz.spacing * 1),
                help_text='<PATH>.svs')
            if changed:
                self.load(self.user_slide, ignore_errors=True)
            if imgui.is_item_hovered() and not imgui.is_item_active() and self.user_slide != '':
                imgui.set_tooltip(self.user_slide)
            imgui.same_line()
            if imgui_utils.button('Browse...', width=viz.button_w, enabled=(viz.project_widget.P is not None)):
                imgui.open_popup('project_slides_popup')

            dim_color = list(imgui.get_style().colors[imgui.COLOR_TEXT])
            dim_color[-1] *= 0.5

            # WSI thumbnail ===================================================
            width = viz.font_size * 28
            height = imgui.get_text_line_height_with_spacing() * 12 + viz.spacing
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
            imgui.push_style_color(imgui.COLOR_HEADER, 0, 0, 0, 0)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.16, 0.29, 0.48, 0.5)
            imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0.16, 0.29, 0.48, 0.9)
            imgui.begin_child('##slide_thumb', width=width, height=height, border=True)

            if viz.wsi_thumb is not None:
                hw_ratio = (viz.wsi_thumb.shape[0] / viz.wsi_thumb.shape[1])
                max_width = min(width - viz.spacing*2, (height - viz.spacing*2) / hw_ratio)
                max_height = max_width * hw_ratio

                if viz._wsi_tex_obj is not None:
                    imgui.same_line(int((width - max_width)/2))
                    imgui.image(viz._wsi_tex_obj.gl_id, max_width, max_height)

                    # Show location overlay
                    if viz.wsi_window_size:
                        # Convert from wsi coords to thumbnail coords
                        t_x, t_y = imgui.get_window_position()
                        t_x = t_x + int((width - max_width)/2)
                        t_w_ratio = max_width / viz.wsi.dimensions[0]
                        t_h_ratio = max_height / viz.wsi.dimensions[1]
                        t_x += viz.thumb_origin[0] * t_w_ratio
                        t_y += viz.thumb_origin[1] * t_h_ratio
                        t_y += viz.spacing

                        draw_list = imgui.get_overlay_draw_list()
                        draw_list.add_rect(
                            t_x,
                            t_y,
                            t_x + (viz.wsi_window_size[0] * t_w_ratio),
                            t_y + (viz.wsi_window_size[1] * t_h_ratio),
                            imgui.get_color_u32_rgba(0, 0, 0, 1),
                            thickness=2)

            if viz.wsi_thumb is None:
                imgui.text_colored('Slide not loaded', *dim_color)

            imgui.end_child()
            imgui.pop_style_color(3)
            imgui.pop_style_var(1)
            # =================================================================

            # Slide options and properties (child) ============================
            imgui.same_line()
            imgui.begin_child('##slide_options', width=-1, height=height, border=False)

            # Slide properties (sub-child). -----------------------------------
            if viz.wsi is not None:
                prop = viz.wsi.properties
                if self._filter_grid is not None and self.show_tile_filter:
                    est_tiles = int(self._filter_grid.sum())
                elif self.show_slide_filter:
                    est_tiles = viz.wsi.estimated_num_tiles
                else:
                    est_tiles = viz.wsi.grid.shape[0] * viz.wsi.grid.shape[1]
                vals = [
                    f"{prop['width']} x {prop['height']}",
                    f'{viz.wsi.mpp:.4f} ({int(10 / (viz.wsi.slide.level_downsamples[0] * viz.wsi.mpp)):d}x)',
                    viz.wsi.vendor if viz.wsi.vendor is not None else '-',
                    str(est_tiles),
                    str(self.num_total_rois)
                ]
            else:
                vals = ["-" for _ in range(8)]
            rows = [
                ['Dimensions (w x h)',  vals[0]],
                ['MPP (Magnification)', vals[1]],
                ['Scanner',             vals[2]],
                ['Est. tiles',          vals[3]],
                ['ROIs',                vals[4]],
            ]
            height = imgui.get_text_line_height_with_spacing() * len(rows) + viz.spacing
            imgui.begin_child('##slide_properties', width=-1, height=height, border=True, flags=imgui.WINDOW_NO_SCROLLBAR)
            for y, cols in enumerate(rows):
                for x, col in enumerate(cols):
                    if x != 0:
                        imgui.same_line(viz.font_size * (8 + (x - 1) * 6))
                    if x == 0:
                        imgui.text_colored(col, *dim_color)
                    else:
                        imgui.text(col)

            with imgui_utils.grayed_out(not viz.wsi or not self.num_total_rois):
                imgui.same_line(imgui.get_content_region_max()[0] - viz.font_size * 4 - viz.spacing * 3)
                _rois_clicked, self._use_rois = imgui.checkbox("Use ROIs", self._use_rois)
                if _rois_clicked:
                    viz._reload_wsi(use_rois=self._use_rois)

            imgui.end_child()
            # -----------------------------------------------------------------

            # Tile filtering
            _filter_clicked, self.show_tile_filter = imgui.checkbox('Tile filter', self.show_tile_filter)
            if _filter_clicked:
                self.update_tile_filter()
            imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*8)
            with imgui_utils.item_width(viz.font_size * 8):
                _stride_changed, _stride = imgui.slider_int('##stride',
                                                            self.stride,
                                                            min_value=1,
                                                            max_value=16,
                                                            format='Stride %d')
                if _stride_changed:
                    self._capturing_stride = _stride
                if imgui.is_mouse_released() and self._capturing_stride:
                    self.stride = self._capturing_stride
                    self._capturing_stride = None
                    self.refresh_stride()

            # Slide filtering
            _qc_clicked, self.show_slide_filter = imgui.checkbox('Slide filter', self.show_slide_filter)
            imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*8)
            with imgui_utils.item_width(viz.font_size * 8), imgui_utils.grayed_out(not self.show_slide_filter):
                _qc_method_clicked, self.qc_idx = imgui.combo("##qc_method", self.qc_idx, self._qc_methods_str)
            if _qc_clicked or _qc_method_clicked:
                self.update_slide_filter()

            # Normalizing
            _norm_clicked, self.normalize_wsi = imgui.checkbox('Normalize', self.normalize_wsi)
            viz._normalize_wsi = self.normalize_wsi
            imgui.same_line(imgui.get_content_region_max()[0] - 1 - viz.font_size*8)
            with imgui_utils.item_width(viz.font_size * 8), imgui_utils.grayed_out(not self.normalize_wsi):
                _norm_method_clicked, self.norm_idx = imgui.combo("##norm_method", self.norm_idx, self._normalizer_methods_str)
            if _norm_clicked or _norm_method_clicked:
                self.change_normalizer()
                viz._refresh_thumb = True

            # Grayspace & whitespace filtering --------------------------------
            with imgui_utils.grayed_out(self._thread_is_running or not self.show_tile_filter):
                imgui.separator()
                imgui.text("Grayspace")
                imgui.same_line(viz.label_w)
                slider_w = (imgui.get_content_region_max()[0] - (viz.spacing + viz.label_w)) / 2
                with imgui_utils.item_width(slider_w):
                    _gsf_changed, _gs_frac = imgui.slider_float('##gs_fraction',
                                                                 self.gs_fraction,
                                                                 min_value=0,
                                                                 max_value=1,
                                                                 format='Fraction %.2f')
                imgui.same_line()
                with imgui_utils.item_width(slider_w):
                    _gst_changed, _gs_thresh = imgui.slider_float('##gs_threshold',
                                                                  self.gs_threshold,
                                                                  min_value=0,
                                                                  max_value=1,
                                                                  format='Thresh %.2f')

                imgui.text("Whitespace")
                imgui.same_line(viz.label_w)
                with imgui_utils.item_width(slider_w):
                    _wsf_changed, _ws_frac = imgui.slider_float('##ws_fraction',
                                                                self.ws_fraction,
                                                                min_value=0,
                                                                max_value=1,
                                                                format='Fraction %.2f')
                imgui.same_line()
                with imgui_utils.item_width(slider_w):
                    _wst_changed, _ws_thresh = imgui.slider_float('##ws_threshold',
                                                                  self.ws_threshold,
                                                                  min_value=0,
                                                                  max_value=255,
                                                                  format='Thresh %.0f')

                if not self._thread_is_running:
                    if _gsf_changed or _wsf_changed:
                        self.gs_fraction = _gs_frac
                        self.ws_fraction = _ws_frac
                        self.refresh_gs_ws()
                    if _gst_changed or _wst_changed:
                        self._capturing_ws_thresh = _ws_thresh
                        self._capturing_gs_thresh = _gs_thresh
                if imgui.is_mouse_released() and self._capturing_gs_thresh:
                    self.gs_threshold = self._capturing_gs_thresh
                    self.ws_threshold = self._capturing_ws_thresh
                    self._capturing_ws_thresh = None
                    self._capturing_gs_thresh = None
                    self.reset_tile_filter_and_join_thread()
                    self.update_tile_filter()

            # -----------------------------------------------------------------

            imgui.end_child()

            # =================================================================

        if imgui.begin_popup('project_slides_popup'):
            if len(self.project_slides) == 0:
                    with imgui_utils.grayed_out():
                        imgui.menu_item('No results found')
            for slide in self.project_slides:
                clicked, _state = imgui.menu_item(slide)
                if clicked:
                    self.load(slide, ignore_errors=True)
            imgui.end_popup()

        paths = viz.pop_drag_and_drop_paths()
        if paths is not None and len(paths) >= 1:
            self.load(paths[0], ignore_errors=True)

    def list_runs_and_slides(self, parents):
        items = []
        run_regex = re.compile(r'\d+-.*')
        slide_regex = re.compile(r'network-snapshot-\d+\.slide')
        for parent in set(parents):
            if os.path.isdir(parent):
                for entry in os.scandir(parent):
                    if entry.is_dir() and run_regex.fullmatch(entry.name):
                        items.append(EasyDict(type='run', name=entry.name, path=os.path.join(parent, entry.name)))
                    if entry.is_file() and slide_regex.fullmatch(entry.name):
                        items.append(EasyDict(type='slide', name=entry.name, path=os.path.join(parent, entry.name)))

        items = sorted(items, key=lambda item: (item.name.replace('_', ' '), item.path))
        return items

#----------------------------------------------------------------------------
