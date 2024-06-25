import rasterio
import uuid
import numpy
import copy

import eoscale.utils as eoutils
import eoscale.shared as eosh
import eoscale.data_types as eodt


class EOContextManager:

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
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end()

    # Private methods

    def _release_all(self):

        self.shared_mem_views = dict()
        self.shared_data_types = dict()

        for key in self.shared_resources:
            self.shared_resources[key].release()

        self.shared_resources = dict()

    # Public methods

    def open_raster(self,
                    raster_path: str) -> None:
        """
            Create an new shared instance from file
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

    def create_image(self, profile: dict) -> str:
        """
            Given a profile with at least the following keys:
            count
            height
            width
            dtype
            this method allocates a shared image and its metadata 
        """
        eoshared_instance = eosh.EOShared()
        eoshared_instance.create_array(profile=profile)
        self.shared_resources[eoshared_instance.virtual_path] = eoshared_instance
        self.shared_data_types[eoshared_instance.virtual_path] = eodt.DataType.RASTER
        return eoshared_instance.virtual_path

    def create_memview(self, key: str, arr_subset: numpy.ndarray, arr_subset_profile: dict) -> str:
        """
            This method allows the developper to indicate a subset memory view of a shared resource he wants to use as input
            of an executor.
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

    def get_profile(self, key: str) -> dict:
        """
            This method returns a profile from the key given by the user.
            This key can be a shared resource key or a memory view key
        """
        if key in self.shared_mem_views:
            return copy.deepcopy(self.shared_mem_views[key][2])
        else:
            return self.shared_resources[key].get_profile()

    def release(self, key: str):
        """
            Release definitely the corresponding shared resource
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

    def write(self, key: str, img_path: str):
        """
            Write the corresponding shared resource to disk
        """
        if key in self.shared_resources:
            profile = self.shared_resources[key].get_profile()
            img_buffer = self.shared_resources[key].get_array(data_type=self.shared_data_types[key])
            with rasterio.open(img_path, "w", **profile) as out_dataset:
                out_dataset.write(img_buffer)
        else:
            print(f"WARNING: the key {key} to write is not known by the context manager")

    def update_profile(self, key: str, profile: dict) -> str:
        """
            This method update the profile of a given key and returns the new key
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

    def start(self):
        if len(self.shared_resources) > 0:
            self._release_all()

    def end(self):
        self._release_all()
