#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2024 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of EOSCale
# (see https://github.com/CNES/eoscale).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import rasterio
import uuid
import numpy
import copy

import eoscale.utils as eoutils
import eoscale.shared as eosh
import eoscale.data_types as eodt


class EOContextManager:
    """
    A context manager for managing shared resources and memory views in an Earth Observation (EO) processing context.

    This class handles the creation, management, and release of shared resources, such as raster images,
    and their associated memory views. It supports multi-processing and tile-based operations.

    Parameters
    ----------
    nb_workers : int
        The number of workers to use for processing.
    tile_mode : bool, optional
        A flag indicating whether tile-based processing mode is enabled (default is False).

    """
    def __init__(self,
                 nb_workers: int,
                 tile_mode: bool = False):

        self.nb_workers = nb_workers
        self.tile_mode = tile_mode
        self.shared_resources: dict = dict()

        # Key is the unique shared resource key and the value is the data type of the shared resources
        self.shared_data_types: dict = dict()

        # Key is a unique memview key and value is a tuple (shared_resource_key, array subset, profile_subset)
        self.shared_mem_views: dict = dict()

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._end()

    def _start(self) -> None:
        if len(self.shared_resources) > 0:
            self._release_all()

    def _end(self) -> None:
        self._release_all()

    def _release_all(self):

        self.shared_mem_views = dict()
        self.shared_data_types = dict()

        for key in self.shared_resources:
            self.shared_resources[key].release()

        self.shared_resources = dict()

    def open_raster(self,
                    raster_path: str) -> eodt.VirtualPath:
        """
        Opens a raster file and registers it as a shared resource.

        Parameters
        ----------
        raster_path : str
            The file path to the raster image to be opened.

        Returns
        -------
        VirtualPath
            The virtual path to the shared raster resource.

        Example
        -------
        >>> virtual_path = open_raster('/path/to/raster/file.tif')
        >>> print(virtual_path)
        '/virtual/path/to/raster/resource'
        """

        new_shared_resource = eosh.EOShared()
        new_shared_resource.create_from_raster_path(raster_path=raster_path)
        self.shared_resources[new_shared_resource.virtual_path] = new_shared_resource
        self.shared_data_types[new_shared_resource.virtual_path] = eodt.DataType.RASTER
        return new_shared_resource.virtual_path

    def open_point_cloud(self,
                         point_cloud_path: str) -> None:

        """
            Create a new shared instance from a point cloud file (readable by laspy) 
        """
        new_shared_resource = eosh.EOShared()
        new_shared_resource.create_from_laspy_point_cloud_path(point_cloud_path=point_cloud_path)
        self.shared_resources[new_shared_resource.virtual_path] = new_shared_resource
        self.shared_data_types[new_shared_resource.virtual_path] = eodt.DataType.POINTCLOUD
        return new_shared_resource.virtual_path

    def create_image(self, profile: rasterio.profiles.Profile) -> str:
        """
        Creates an image full of zeros with the specified raster profile and manages shared resources.
        The image generated can be accessed from any filter

        Parameters
        ----------
        profile : rasterio.profiles.Profile
            A dictionary containing the raster profile specifications.
            This includes details such as dimensions, data type, and other metadata necessary
            for creating the raster array.

        Returns
        -------
        str
            The virtual path to the created raster image.

        Example
        -------
        >>> profile = {
        ...     'width': 512,
        ...     'height': 512,
        ...     'dtype': 'uint8',
        ...     'count': 1,
        ...     'driver': 'GTiff'
        ... }
        >>> virtual_path = create_image(profile)
        >>> vp_path_list = n_images_to_m_images_filter(inputs=[virtual_path],...)
        """
        eoshared_instance = eosh.EOShared()
        eoshared_instance.create_array(profile=profile)
        self.shared_resources[eoshared_instance.virtual_path] = eoshared_instance
        self.shared_data_types[eoshared_instance.virtual_path] = eodt.DataType.RASTER
        return eoshared_instance.virtual_path

    def create_memview(self, key: str, arr_subset: numpy.ndarray, arr_subset_profile: dict) -> str:
        """
        Creates a memory view of a subset of a shared resource for use as input to an executor.

        Parameters
        ----------
        key : str
            The key to associate with the shared resource.
        arr_subset : numpy.ndarray
            The subset of the array to be used as the memory view.
        arr_subset_profile : dict
            The profile dictionary containing metadata for the subset array.

        Returns
        -------
        str
            A unique key for the created memory view.

        Example
        -------
        >>> raster_file = 'my_raster.tif'
        >>> with rasterio.open(raster_file, "r") as raster_dataset:
        >>>     profile = raster_dataset.profile
        >>>     data = raster_dataset.read()
        >>>
        >>> with EOContextManager(nb_workers=4, tile_mode=True) as eoscale_manager:
        >>>     access_key = "some_access_key"
        >>>     new_key = eoscale_manager.create_memview(key=access_key, arr_subset=data, arr_subset_profile=profile)
        >>>     arr_from_memview = eoscale_manager.get_array(new_key)
        """
        mem_view_key: str = str(uuid.uuid4())
        self.shared_mem_views[mem_view_key] = (key, arr_subset, arr_subset_profile)
        return mem_view_key

    def get_array(self, key: str, tile: eoutils.MpTile = None) -> numpy.ndarray:
        """
        Returns a memory view or an array from the key given by the user.

        Parameters
        ----------
        key : str
            A key that identifies the shared resource or memory view to be returned.
        tile : eoutils.MpTile, optional
            An optional tile object specifying the portion of the array to be returned.
            If not provided, the full array is returned.

        Returns
        -------
        numpy.ndarray
            The array or memory view corresponding to the provided key.
            If a tile is specified, returns the subset of the array defined by the tile.

        Raises
        ------
        TypeError
            If the key parameter is not of type 'str'.

        Notes
        -----
        - If the key corresponds to a shared memory view, and no tile is specified,
          the entire memory view is returned.
        - If the key corresponds to a shared memory view and a tile is specified,
          the method returns the portion of the memory view defined by the tile.
        - If the key does not correspond to a shared memory view,
          the method retrieves the array from shared resources using the specified key
          and tile, and the associated data type.

        Warning
        -------
        Users should be aware that the returned array is a view.
        Attempting to access this view outside the EOScale context manager may lead to a
        segmentation fault or a memory leak.

        """
        if not isinstance(key, str):
            raise TypeError(f"key parameters must be type 'str' not '{type(key).__name__}'")
        if key in self.shared_mem_views:
            if tile is None:
                return self.shared_mem_views[key][1]
            else:
                start_y = tile.start_y - tile.top_margin
                end_y = tile.end_y + tile.bottom_margin + 1
                start_x = tile.start_x - tile.left_margin
                end_x = tile.end_x + tile.right_margin + 1
                return self.shared_mem_views[key][1][:, start_y:end_y, start_x:end_x]
        else:
            return self.shared_resources[key].get_array(tile=tile,
                                                        data_type=self.shared_data_types[key])

    def get_profile(self, key: str) -> rasterio.profiles.Profile:
        """
        Retrieves the raster profile associated with a given key.

        Parameters
        ----------
        key : str
            The key associated with the raster resource whose profile is to be retrieved.

        Returns
        -------
        Profile
            A dictionary containing the raster profile specifications.

        Raises
        ------
        KeyError
            If the key does not exist.

        Example
        -------
        >>> profile = get_profile('some_key')
        >>> print(profile)
        {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'width': 512,
            'height': 512,
            'count': 1,
            'crs': None,
            'transform': Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        }
        """
        if key in self.shared_mem_views:
            return copy.deepcopy(self.shared_mem_views[key][2])
        else:
            return self.shared_resources[key].get_profile()

    def release(self, key: str) -> None:
        """
        Releases the shared resource and removes all associated memory views.

        Parameters:
        -----------
        key : str
            The key associated with the shared resource to be released.
        """
        mem_view_keys_to_remove: list = []
        # Remove from the mem view dictionnary all the key related to the share resource key
        for k in self.shared_mem_views:
            if self.shared_mem_views[k][0] == key:
                mem_view_keys_to_remove.append(k)
        for k in mem_view_keys_to_remove:
            del self.shared_mem_views[k]

        if key in self.shared_resources:
            self.shared_resources[key].release()
            del self.shared_resources[key]

        del self.shared_data_types[key]

    def write(self, key: str, img_path: str) -> None:
        """
        Writes the raster data associated with the given key to a specified file path.

        Parameters
        ----------
        key : str
            The key associated with the shared raster resource to be written.
        img_path : str
            The file path where the raster image will be written.

        Example
        -------
        >>> write('some_key', '/path/to/output/file.tif')
        """
        if key in self.shared_resources:
            profile = self.shared_resources[key].get_profile()
            img_buffer = self.shared_resources[key].get_array(data_type=self.shared_data_types[key])
            with rasterio.open(img_path, "w", **profile) as out_dataset:
                out_dataset.write(img_buffer)
        else:
            warnings.warn(f"WARNING: the key {key} to write is not known by the context manager")

    def update_profile(self, key: str, profile: rasterio.profiles.Profile) -> str:
        """
        Updates the raster profile associated with a given key and returns the new key.

        Parameters
        ----------
        key : str
            The key associated with the shared raster resource to be updated.
        profile : Profile
            A dictionary containing the new raster profile specifications.

        Returns
        -------
        str
        The new key associated with the updated shared raster resource.
        """
        tmp_value = self.shared_resources[key]
        tmp_data_type = self.shared_data_types[key]
        del self.shared_resources[key]
        del self.shared_data_types[key]
        tmp_value._release_profile()
        new_key: str = tmp_value._update_profile(profile)
        self.shared_resources[new_key] = tmp_value
        self.shared_data_types[new_key] = tmp_data_type
        return new_key

