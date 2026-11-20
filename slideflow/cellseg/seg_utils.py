import os
import cv2
import numpy as np
import multiprocessing as mp
import slideflow as sf
from functools import partial
from os.path import dirname, join, exists, isfile
from typing import Dict, Optional
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


def sparse_split_indices(shape: int, splits: int):
    if shape <= 1:
        return []
    step = max(1, int(shape / max(1, splits)))
    ar = np.arange(1, shape, step)
    if ar[-1] != shape:
        ar = np.append(ar, shape)
    return list(zip(ar[:-1], ar[1:]))


def get_sparse_chunk_centroid(sparse_mask, shape):
    return np.array([
        np.mean(np.unravel_index(row.data, shape), 1).astype(np.int32)
        if row.getnnz()
        else (0, 0)
        for row in sparse_mask
    ])


def get_sparse_centroid(mask, sparse_mask):
    if sparse_mask.shape[0] <= 1:
        return np.zeros((0, 2), dtype=np.int32)
    n_proc = min(sf.util.num_cpu(default=8), max(1, sparse_mask.shape[0] - 1))
    indices = sparse_split_indices(sparse_mask.shape[0], n_proc)
    if n_proc == 1 or len(indices) <= 1:
        return get_sparse_chunk_centroid(sparse_mask[1:], mask.shape)
    with mp.Pool(n_proc) as pool:
        return np.concatenate(
            pool.map(
                partial(get_sparse_chunk_centroid, shape=mask.shape),
                [sparse_mask[i:j] for (i, j) in indices]
            )
        )


def sparse_mask(mask):
    from scipy.sparse import csr_matrix
    raveled = np.ravel(mask)
    nonzero = raveled > 0
    if not np.any(nonzero):
        return csr_matrix((1, 1), dtype=np.uint32)
    cols = np.where(nonzero)[0]
    return csr_matrix(
        (cols, (raveled[nonzero], cols)),
        shape=(int(mask.max()) + 1, int(cols[-1]) + 1),
        dtype=(np.uint32 if (cols[-1] + 1) < 4294967295 else np.uint64)
    )


# --- WSI Cell Segmentation transformation utility ----------------------------

class ApplySegmentation:
    def __init__(self, source: Optional[str] = None) -> None:
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


CellSegment = ApplySegmentation
