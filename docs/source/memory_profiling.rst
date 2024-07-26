Memory profiling
================

The purpose of this documentation is to inform users about EOScale data management and its persistence in memory.
To do this, we will start from the last point of the tutorial :doc:`water_mask <notebooks/water_mask>`.

To monitor changes in memory usage by the various functions, we're going to use the `memory_profiler <https://pypi.org/project/memory-profiler/>`_ tool.
We're also going to slightly modify the last part of the code to encapsulate it in a python function on
which we'll be able to use memory_profiler's ``profile`` decorator.

The last part of the script become

.. code-block:: python

    @profile
    def main():
        water_mask_agg = Path(OUTPUT_DIR) / "water_mask_agg.tif"
        with EOContextManager(nb_workers=2, tile_mode=True) as eoscale_manager:
            ndwi_keys = compute_ndwi(eoscale_manager, [s2_20180621, s2_20180701, s2_20180706, s2_20180711])
            water_mask_keys = water_mask(eoscale_manager, ndwi_keys, 0)
            water_mask_aggregation_keys = water_mask_aggregation(eoscale_manager, water_mask_keys)
            eoscale_manager.write(water_mask_aggregation_keys[0], str(water_mask_agg))
            print("this is the last line handled by the context manager")

    main()

run the following command to start memory profiling

.. code-block:: bash

    mprof run --multiprocess my_python_script.py

which will prompt in your terminal something as

.. code-block:: bash

    Line #    Mem usage    Increment  Occurrences   Line Contents
    =============================================================
       213     98.9 MiB     98.9 MiB           1   @profile
       214                                         def main():
       215     98.9 MiB      0.0 MiB           1       water_mask_agg = Path(OUTPUT_DIR) / "water_mask_agg.tif"
       216     98.9 MiB      0.0 MiB           1       with EOContextManager(nb_workers=2, tile_mode=True) as eoscale_manager:
       217    863.8 MiB    764.8 MiB           1           ndwi_keys = compute_ndwi(eoscale_manager, [s2_20180621, s2_20180701, s2_20180706, s2_20180711])
       218   1174.2 MiB    310.5 MiB           1           water_mask_keys = water_mask(eoscale_manager, ndwi_keys, 0)
       219   1250.7 MiB     76.5 MiB           1           water_mask_aggregation_keys = water_mask_aggregation(eoscale_manager, water_mask_keys)
       220   1251.4 MiB      0.6 MiB           1           eoscale_manager.write(water_mask_aggregation_keys[0], str(water_mask_agg))
       221    257.0 MiB   -994.4 MiB           1           print("this is the last line handled by the context manager")

As you can see, the maximum memory usage is 1.2 GB. It is important to note that it is the ``compute_ndwi`` function
that adds the most data to memory, but it is also this function that is responsible for reading Sentinel-2 images to feed EOScale.
It is also important to note that each call to :func:`eoscale.core.eo_executors.n_images_to_m_images_filter`. via the EOScale
:doc:`available filters <module_filters>` or your own filters will generate numpy arrays in memory which will
be kept until the context manager is closed.


.. note::

    As EOScale stores all the data manipulated in memory, it is important to choose the type of these
    variables carefully. It is also possible to delete certain variables that have become useless in
    the context manage using :meth:`eoscale.core.manager.EOContextManager.release`.