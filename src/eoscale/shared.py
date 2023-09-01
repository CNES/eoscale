from multiprocessing import shared_memory
import rasterio
import numpy
import uuid
import os
import json
import copy
import eoscale.utils as eoutils

EOSHARED_PREFIX: str = "eoshared"
EOSHARED_MTD: str = "metadata"

class EOShared:

    def __init__(self, virtual_path: str = None):
        """ """
        self.shared_array_memory = None
        self.shared_metadata_memory = None
        self.virtual_path: str = None

        if virtual_path is not None:
            self._open_from_virtual_path(virtual_path = virtual_path)
    
    def _extract_from_vpath(self) -> tuple:
        """
            Extract resource key and metada length from the virtual path
        """
        split_v_path: list = self.virtual_path.split("/")
        resource_key: str = split_v_path[2]
        mtd_len: str = split_v_path[1]
        return resource_key, mtd_len    

    def _open_from_virtual_path(self, virtual_path: str):
        """ """

        self.virtual_path = virtual_path
        
        resource_key, mtd_len = self._extract_from_vpath()

        self.shared_array_memory = shared_memory.SharedMemory(name=resource_key, 
                                                              create=False)

        self.shared_metadata_memory = shared_memory.SharedMemory(name=resource_key + EOSHARED_MTD, 
                                                                 create=False)

    def _build_virtual_path(self, key: str, mtd_len: str) -> None:
        self.virtual_path = EOSHARED_PREFIX + "/" + mtd_len + "/" + key
    

    def create_array(self, profile: dict):
        """
            Allocate array 
        """
        # Shared key is made unique
        # this property is awesome since it allows the communication between parallel tasks
        resource_key: str = str(uuid.uuid4())

        # Compute the number of bytes of this array
        d_size = numpy.dtype(profile["dtype"]).itemsize * profile["count"] * profile["height"] * profile["width"]

        # Create a shared memory instance of it
        # shared memory must remain open to keep the memory view
        self.shared_array_memory = shared_memory.SharedMemory(create=True, 
                                                              size=d_size, 
                                                              name=resource_key)

        # Encode and compute the number of bytes of the metadata
        encoded_metadata = json.dumps(eoutils.rasterio_profile_to_dict(profile)).encode()
        mtd_size: int = len(encoded_metadata)
        self.shared_metadata_memory = shared_memory.SharedMemory(create=True, 
                                                                 size=mtd_size, 
                                                                 name=resource_key + EOSHARED_MTD)
        self.shared_metadata_memory.buf[:] = encoded_metadata[:]

        # Create the virtual path to these shared resources
        self._build_virtual_path(mtd_len=str(mtd_size), key = resource_key)


    def create_from_raster_path(self,
                                raster_path: str) -> str :
        
        """ Create a shared memory numpy array from a raster image """
        
        with rasterio.open(raster_path, "r") as raster_dataset:

            # Shared key is made unique
            # this property is awesome since it allows the communication between parallel tasks
            resource_key: str = str(uuid.uuid4())

            # Compute the number of bytes of this array
            d_size = numpy.dtype(raster_dataset.dtypes[0]).itemsize * raster_dataset.count * raster_dataset.height * raster_dataset.width

            # Create a shared memory instance of it
            # shared memory must remain open to keep the memory view
            self.shared_array_memory = shared_memory.SharedMemory(create=True, 
                                                                  size=d_size, 
                                                                  name=resource_key)
            

            big_array = numpy.ndarray(shape=(raster_dataset.count  * raster_dataset.height * raster_dataset.width), 
                                      dtype=raster_dataset.dtypes[0], 
                                      buffer=self.shared_array_memory.buf)

            big_array[:] = raster_dataset.read().flatten()[:]

            # Encode and compute the number of bytes of the metadata
            encoded_metadata = json.dumps(eoutils.rasterio_profile_to_dict(raster_dataset.profile)).encode()
            mtd_size: int = len(encoded_metadata)
            self.shared_metadata_memory = shared_memory.SharedMemory(create=True, 
                                                                     size=mtd_size, 
                                                                     name=resource_key + EOSHARED_MTD)
            self.shared_metadata_memory.buf[:] = encoded_metadata[:]

            # Create the virtual path to these shared resources
            self._build_virtual_path(mtd_len=str(mtd_size), key = resource_key)

    def get_profile(self) -> rasterio.DatasetReader.profile:
        """
            Return a copy of the rasterio profile
        """
        resource_key, mtd_len = self._extract_from_vpath()
        encoded_mtd = bytearray(int(mtd_len))
        encoded_mtd[:] = self.shared_metadata_memory.buf[:]
        return copy.deepcopy(eoutils.dict_to_rasterio_profile(json.loads(encoded_mtd.decode())))

    def get_array(self, 
                  tile: eoutils.MpTile = None) -> numpy.ndarray:

        """            
            Return a memory view of the array or a subset of it if a tile is given
        """
        profile = self.get_profile()
        array_shape = (profile['count'], profile['height'], profile['width'])

        if tile is None:
            return numpy.ndarray(array_shape,
                                 dtype=profile['dtype'],
                                 buffer=self.shared_array_memory.buf)
        else:
            start_y = tile.start_y - tile.top_margin
            end_y = tile.end_y + tile.bottom_margin + 1
            start_x = tile.start_x - tile.left_margin
            end_x = tile.end_x + tile.right_margin + 1
            return numpy.ndarray(array_shape,
                                 dtype=metadata['dtype'],
                                 buffer=self.shared_array_memory.buf)[:, start_y:end_y, start_x:end_x]
    
    def close(self):
        """ 
            A close does not mean release from memory. Must be called by a process once it has finished
            with this resource.
        """
        if self.shared_array_memory is not None:
            self.shared_array_memory.close()
        
        if self.shared_metadata_memory is not None:
            self.shared_metadata_memory.close()
    
    def release(self):
    
        """ 
            Definitely release the shared memory.
        """
        if self.shared_array_memory is not None:
            self.shared_array_memory.close()
            self.shared_array_memory.unlink()
        
        if self.shared_metadata_memory is not None:
            self.shared_metadata_memory.close()
            self.shared_metadata_memory.unlink()

# class EOShared:

#     def __init__(self, eoshared_vpath: str = None):

#         self.shared_array_memory = None
#         self.shared_metadata_memory = None

#         # One virtual path for tracing this resource (array & metadata)
#         # EOSHARED_PREFIX / MTD_LEN / RESOURCE_KEY
#         self.eoshared_vpath: str = None
        
#         # Id of the process that creates this shared resource
#         self.pid_creator: int = None

#         if eoshared_vpath is not None:
#             self.create_from_existing(eoshared_vpath = eoshared_vpath)
    
#     def build_vpath(self, mtd_len: str, resource_key: str):
#         """
#             Build a virtual path composed of a prefix to identify that this resource is shared among processes,
#             the byte length of the metadata and an unique identifier to retrieve the shared memory bloc.
#         """
#         self.eoshared_vpath = EOSHARED_PREFIX + "/" + mtd_len + "/" + resource_key
    
#     def extract_from_vpath(self) -> tuple:
#         """
#             Extract resource key and metada length from the virtual path
#         """
#         eoshared_vpath_split: list = self.eoshared_vpath.split("/")
#         resource_key: str = eoshared_vpath_split[2]
#         mtd_len: str = eoshared_vpath_split[1]
#         return resource_key, mtd_len

#     def create_from_existing(self, eoshared_vpath: str):
#         """ Must not be called by a user directly """
        
#         self.eoshared_vpath = eoshared_vpath
        
#         resource_key, mtd_len = self.extract_from_vpath()

#         self.shared_array_memory = shared_memory.SharedMemory(name=resource_key, 
#                                                               create=False)

#         self.shared_metadata_memory = shared_memory.SharedMemory(name=resource_key + EOSHARED_MTD, 
#                                                                  create=False)

#     def create_from_image_path(self, image_path: str):

#         """ Must not be called by a user directly """
#         with rasterio.open(image_path, "r") as img_dataset:

#             # Shared key is made unique
#             # this property is awesome since it allows the communication between parallel tasks
#             resource_key: str = str(uuid.uuid4())

#             # Compute the number of bytes of this array
#             d_size = numpy.dtype(img_dataset.dtypes[0]).itemsize * img_dataset.count * img_dataset.height * img_dataset.width
            
#             # Create a shared memory instance of it
#             # shared memory must remain open to keep the memory view
#             self.shared_array_memory = shared_memory.SharedMemory(create=True, 
#                                                                   size=d_size, 
#                                                                   name=resource_key)

#             big_array = numpy.ndarray(shape=(img_dataset.count  * img_dataset.height * img_dataset.width), dtype=img_dataset.dtypes[0], buffer=self.shared_array_memory.buf)
#             big_array[:] = img_dataset.read().flatten()[:]
            
#             # Encode and compute the number of bytes of the metadata
#             encoded_metadata = json.dumps(eoutils.rasterio_profile_to_dict(img_dataset.profile)).encode()
#             mtd_size: int = len(encoded_metadata)
#             self.shared_metadata_memory = shared_memory.SharedMemory(create=True, 
#                                                                      size=mtd_size, 
#                                                                      name=resource_key + EOSHARED_MTD)
#             self.shared_metadata_memory.buf[:] = encoded_metadata[:]

#             # Assign this share resource to this process (it is the master process in general)
#             self.pid_creator = os.getpid()

#             # Create the virtual path to these shared resources
#             self.build_vpath(mtd_len=str(mtd_size), resource_key = resource_key)
            
    
#     def create_from_in_memory_array(self, 
#                                     big_array: numpy.ndarray, 
#                                     user_metadata: dict = None):
#         """
#             Given a big array (resulting from a previous multiprocessing filter), this method shares it for the
#             subsequent multiprocessed filters.
#             No overload of memory !
#             Concerning the metadata, the user can give a profile similar to rasterio
#         """
        
#         # Shared key is made unique
#         resource_key: str = str(uuid.uuid4())

#         encoded_metadata = None
#         count: int = None
#         height: int = None
#         width: int = None

#         if not ("count" in user_metadata and 
#                 "height" in user_metadata and
#                 "width" in user_metadata ):
#             raise ValueError("Custom metadata must contain at least the following keys (count, height, width)")

#         count = user_metadata['count']
#         height = user_metadata["height"]
#         width = user_metadata["width"]
#         encoded_metadata = json.dumps(eoutils.rasterio_profile_to_dict(user_metadata)).encode()
        
#         # Create the shared array memory from big array
#         d_size = numpy.dtype(big_array.dtype).itemsize * count * height * width
#         self.shared_array_memory = shared_memory.SharedMemory(create=True, 
#                                                               size=d_size, 
#                                                               name=resource_key)
        
#         shared_big_array = numpy.ndarray( shape = (count * height * width), 
#                                           dtype = big_array.dtype,
#                                           buffer = self.shared_array_memory.buf)
#         shared_big_array[:] = big_array.flatten()[:]

#         # Create the shared metadata buffer        
#         mtd_size: int = len(encoded_metadata)
#         self.shared_metadata_memory = shared_memory.SharedMemory(create=True, 
#                                                                  size=mtd_size, 
#                                                                  name=resource_key + EOSHARED_MTD)
#         self.shared_metadata_memory.buf[:] = encoded_metadata[:]

#         # Assign this share resource to this process (it is the master process in general)
#         self.pid_creator = os.getpid()

#         # Create the virtual path to these shared resources
#         self.build_vpath(mtd_len=str(mtd_size), resource_key = resource_key)

    
#     def get_metadata(self) -> rasterio.DatasetReader.profile:
#         """
#             Return the metadata
#         """
#         resource_key, mtd_len = self.extract_from_vpath()
#         encoded_mtd = bytearray(int(mtd_len))
#         encoded_mtd[:] = self.shared_metadata_memory.buf[:]
#         return eoutils.dict_to_rasterio_profile(json.loads(encoded_mtd.decode()))
    
#     def get_array(self, strip: eoutils.MpStrip = None) -> numpy.ndarray:
#         """
#             Return a memory view of the array or a subset of it if a strip is given
#         """
#         metadata = self.get_metadata()
#         array_shape = (metadata['count'], metadata['height'], metadata['width'])
#         if strip is None:
#             return numpy.ndarray(array_shape,
#                                  dtype=metadata['dtype'],
#                                  buffer=self.shared_array_memory.buf)
#         else:
#             start_y = strip.start_y - strip.top_margin
#             end_y = strip.end_y + strip.bottom_margin + 1
#             return numpy.ndarray(array_shape,
#                                  dtype=metadata['dtype'],
#                                  buffer=self.shared_array_memory.buf)[:, start_y:end_y, :]
    
#     def get_vpath(self) -> str:
#         """ 
#             Return the virtual path of this resource.
#             Can be used as input of a subsequent multiprocessed algorithm.
#         """
#         return self.eoshared_vpath
    
#     def close(self):
#         """ 
#             A close does not mean release from memory. Must be called by a process once it has finished
#             with this resource.
#         """
#         self.shared_array_memory.close()
#         self.shared_metadata_memory.close()

#     def release(self):
#         """ 
#             This method must be called by the process responsible of the
#             creation of this shared resource.
#         """
#         if self.shared_array_memory is not None:
#             self.shared_array_memory.close()
#             self.shared_array_memory.unlink()
        
#         if self.shared_metadata_memory is not None:
#             self.shared_metadata_memory.close()
#             self.shared_metadata_memory.unlink()

# class ShOpen(object):
#     """
#         This allows to create a shared image resource and make it available
#         within its scope.
#     """

#     def __init__(self, resource_path: str):
#         self.resource_path = resource_path
#         self.eo_shared = EOShared()

#     def __enter__(self):
#         """ """
#         if self.resource_path.startswith(EOSHARED_PREFIX):
#             # Create a shared memory instance from it
#             self.eo_shared.create_from_existing(self.resource_path)
#         else:
#             self.eo_shared.create_from_image_path(self.resource_path)
        
#         return self.eo_shared.get_vpath()
    
#     def __exit__(self, exc_type, exc_value, traceback):
#         """
#             Giving the role to the class ShOpen to destroy the
#             shared resource ensures that it is the same process
#             that creates and release this resource. 
#         """
#         if self.eo_shared.pid_creator is not None and os.getpid() == self.eo_shared.pid_creator:
#             self.eo_shared.release()
#         else:
#             self.eo_shared.close()

# class ShMake(object):
#     """
#         This allows to create 
#     """
#     def __init__(self, 
#                  big_array: numpy.ndarray, 
#                  user_metadata: dict = None,
#                  img_dataset: rasterio.DatasetReader = None):
#         """
#             metadata must at least contain the following keys:
#             count
#             height
#             width
#             dtype
#         """
#         self.big_array = big_array
#         self.user_metadata =  user_metadata
#         self.img_dataset = img_dataset
#         self.eo_shared = EOShared()
    
#     def __enter__(self):
#         """ """
#         # Release resources so that the user
#         # car release the big array from memory
#         self.eo_shared.create_from_in_memory_array(big_array = self.big_array,
#                                                    user_metadata = self.user_metadata,
#                                                    img_dataset = self.img_dataset)

#         self.big_array = None
#         self.metadata = None

#         return self.eo_shared.get_vpath()
    
#     def __exit__(self, exc_type, exc_value, traceback):
#         """
#             Systematically release the shared memory
#         """
#         self.eo_shared.release()
