import os
import cv2
import numpy as np
import multiprocessing as mp
import slideflow as sf
from os.path import dirname, join, exists, isfile
from typing import Any, Dict, Optional
import zarr

_ZARR_V3 = int(zarr.__version__.split('.')[0]) >= 3

# --- Utility functions -------------------------------------------------------


def _open_store(store, mode: str):
    """Open a Zarr store, automatically wrapping .zip paths in ZipStore across Zarr v2 & v3."""
    if isinstance(store, (str, os.PathLike)):
        store_str = os.fspath(store)
        if store_str.endswith('.zip'):
            if _ZARR_V3 and hasattr(zarr, 'storage') and hasattr(zarr.storage, 'ZipStore'):
                return zarr.storage.ZipStore(store_str, mode=mode), True
            return zarr.ZipStore(store_str, mode=mode), True
        return store_str, False
    return store, False


def save_zarr_compressed(
    store,
    *args,
    zarr_version: Optional[int] = None,
    path: Optional[str] = None,
    compressor=None,
    **kwargs
) -> None:
    """
    Save one or more arrays into a new Zarr group (compatible with Zarr v2 and v3).

    Parameters
    ----------
    store : Store or str
        A Zarr store or a path (string or os.PathLike) where the Zarr data will be written.
    *args : arrays
        Positional arrays to be stored. They will be saved with keys "arr_0", "arr_1", etc.
    zarr_version : int, optional
        Zarr format version (defaults to 2 for cross-version numcodecs compatibility).
    path : str, optional
        Subpath within the store.
    compressor : numcodecs.Codec, optional
        A compressor instance to use.
    **kwargs : arrays
        Keyword arrays to store (with the key names as provided).

    Raises
    ------
    ValueError
        If no array is provided.
    """
    if len(args) == 0 and len(kwargs) == 0:
        raise ValueError('At least one array must be provided')

    target_store, should_close = _open_store(store, mode='w')
    fmt = zarr_version if zarr_version is not None else 2
    try:
        if _ZARR_V3:
            grp = zarr.open_group(target_store, mode='w', path=path, zarr_format=fmt)
            comp_kw = {'compressor': compressor} if compressor is not None else {}
            for i, arr in enumerate(args):
                try:
                    grp.create_array(name=f'arr_{i}', data=np.asarray(arr), overwrite=True, **comp_kw)
                except (TypeError, ValueError):
                    grp.create_array(name=f'arr_{i}', data=np.asarray(arr), overwrite=True)
            for key, arr in kwargs.items():
                try:
                    grp.create_array(name=key, data=np.asarray(arr), overwrite=True, **comp_kw)
                except (TypeError, ValueError):
                    grp.create_array(name=key, data=np.asarray(arr), overwrite=True)
        else:
            grp = zarr.open_group(target_store, mode='w', path=path)
            for i, arr in enumerate(args):
                grp.create_dataset(f'arr_{i}', data=np.asarray(arr), overwrite=True, compressor=compressor)
            for key, arr in kwargs.items():
                grp.create_dataset(key, data=np.asarray(arr), overwrite=True, compressor=compressor)
    finally:
        if should_close and hasattr(target_store, 'close'):
            target_store.close()
        elif hasattr(store, 'close') and not isinstance(store, (str, os.PathLike)):
            store.close()


def load_zarr_arrays(store, path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load all arrays from a Zarr group or .zip store into a dict (Zarr v2 & v3 compatible)."""
    target_store, should_close = _open_store(store, mode='r')
    try:
        grp = zarr.open_group(target_store, mode='r', path=path)
        return {key: np.asarray(grp[key][...]) for key in grp.keys()}
    finally:
        if should_close and hasattr(target_store, 'close'):
            target_store.close()


def fast_outlines_list(masks, num_threads: Optional[int] = None):
    """
    Get outlines of mask regions as a list for plotting.
    Uses multithreading to process large images.
    """
    if num_threads is None:
        num_threads = sf.util.num_cpu()

    def get_mask_outline(mask_id):
        mn = (masks == mask_id)
        if mn.sum() > 0:
            contours = cv2.findContours(
                mn.astype(np.uint8),
                mode=cv2.RETR_EXTERNAL,
                method=cv2.CHAIN_APPROX_NONE
            )[-2]
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix) > 4:
                return pix
            else:
                return np.zeros((0, 2), dtype=int)
        return np.zeros((0, 2), dtype=int)

    with mp.dummy.Pool(num_threads) as pool:
        return pool.map(get_mask_outline, np.unique(masks)[1:])


# --- WSI QC classes ----------------------------------------------------------

class CellSegment:
    def __init__(self, source: Optional[str] = None):
        """
        QC function which loads a saved numpy mask to a WSI.

        If source is not specified, searches in the same directory as the slide.

        Parameters
        ----------
        source : str, optional
            Path to search for the QC mask. This may be a *.zip file or a directory.
        """
        self.source = source

    def __repr__(self):
        return f"CellSegment(source={self.source!r})"

    def __call__(self, wsi: "sf.WSI") -> None:
        """
        Applies a segmentation mask to a given slide.

        Parameters
        ----------
        wsi : sf.WSI
            Whole-slide image.

        Raises
        ------
        sf.errors.QCError
            If a segmentation mask is not found.
        """
        from slideflow.cellseg import Segmentation

        # Determine the source path (if not provided, use the slide's directory)
        source = self.source if self.source is not None else dirname(wsi.path)

        if exists(source) and isfile(source):
            mask_path = source
        elif exists(join(source, wsi.name + '-masks.zip')):
            mask_path = join(source, wsi.name + '-masks.zip')
        else:
            raise sf.errors.QCError("Segmentation mask not found.")
        seg = Segmentation.load(mask_path)
        wsi.apply_segmentation(seg)
        return None
