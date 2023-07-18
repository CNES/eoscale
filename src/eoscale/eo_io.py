import rasterio
import eoscale.shared as eosh

def write_images(output_paths: list = None,
                 eo_shared_instances: list = None,
                 input_vpaths: list = None):
    """ 
        Given input eoshared instances or virtual paths, this methods
        writes them to to disk given the output paths.
    """
    if eo_shared_instances is not None:
        if len(output_paths) != len(eo_shared_instances):
            raise ValueError("Size of output paths and eo_shared_instances are not the same")
        for i in range(len(eo_shared_instances)):
            array = eo_shared_instances[i].get_array()
            metadata = eo_shared_instances[i].get_metadata()
            with rasterio.open( output_paths[i], "w", **metadata) as out_img_dataset:
                out_img_dataset.write(array)
    elif input_vpaths is not None:
        if len(output_paths) != len(input_vpaths):
            raise ValueError("Size of output paths and input_vpaths are not the same")
        for i in range(len(input_vpaths)):
            eo_shared_inst = eosh.EOShared(eoshared_vpath=input_vpaths[i])
            array = eo_shared_inst.get_array()
            metadata = eo_shared_inst.get_metadata()
            with rasterio.open( output_paths[i], "w", **metadata) as out_img_dataset:
                out_img_dataset.write(array)
            eo_shared_inst.close()