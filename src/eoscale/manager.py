import eoscale.shared as eosh

class EOContextManager:

    def __init__(self, 
                 nb_workers:int,
                 tile_mode: bool = False):

        self.nb_workers = nb_workers
        self.tile_mode = tile_mode
        self.shared_resources: dict = dict()
    
    # Private methods

    def _release_all(self):

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
        new_shared_resource.create_from_raster_path(raster_path = raster_path)
        self.shared_resources[new_shared_resource.virtual_path] = new_shared_resource
        return new_shared_resource.virtual_path
    
    def start(self):
        if len(self.shared_resources) > 0:
            self._release_all()

    def end(self):
        self._release_all()