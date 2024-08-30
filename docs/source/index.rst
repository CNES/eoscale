.. EOScale documentation master file, created by
   sphinx-quickstart on Wed Jun 26 10:36:53 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EOScale's documentation!
===================================

EOScale's purpose
*****************

A situation we have come accross very frequently as remote sensing engineer at CNES is the need to keep one or multiple NumPy arrays in memory for processing large satellite images in a parallel environnement that can be distributed or not.

Because Python multithreading is not simultaneous and not designed for maximising the use of CPUs `(I recommend this nice youtube video) <https://www.youtube.com/watch?v=AZnGRKFUU0c>`_, we choose Python multiprocessing module for scaling our algorithms.

However, having to use multiple processes means some complexity when it comes to sharing easily large image rasters and their metadata. Fortunately since Python 3.8+, the concept of shared_memory has been introduced to share data structures between processes. It relies on posix mmap2 under the hood.

EOScale relies on this concept to store and share large satellite images and their metadatas between processes without duplicating memory space.

Currently, EOScale provides 2 paradigms:

- A generic N image to M image filter that uses a tiling strategy with the concept of stability margin to parallelize local dependency algorithms while ensuring identical results. All the complexity is done for you, you just have to define your algorithm as a callable function that takes as input a list of numpy arrays, a list of their corresponding image metadata and your filter parameters as a Python dictionnary and that is all !

- A generic N image to M scalars that can returns anything that can be concatenated in a Map/Reduce paradigm. For example a histogram or statiscal values such as min, max or average.

Your pipeline in memory
***********************

One other great advantage of EOScale is how easy it is to chain your filters through your pipeline **in memory** and again while minimizing your memory footprint. This allows your programs to be more efficient and less consuming regarding your energy footprint.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Tutorials <tutorials>
   Available Filters <module_filters>
   API <advanced_uses>
   Contributing <contributing>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`