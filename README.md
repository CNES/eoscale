# EOSCALE

## What is the purpose ?

A situation we have come accross very frequently as remote sensing engineer at CNES is the need to keep one or multiple NumPy arrays in memory for processing large satellite images in a parallel environnement that can be distributed or not.

Because Python multithreading is not simultaneous and not designed for maximising the use of CPUs ([I recommend this nice youtube video](https://www.youtube.com/watch?v=AZnGRKFUU0c)), we choose Python multiprocessing module for scaling our algorithms.

However, having to use multiple processes means some complexity when it comes to sharing easily large image rasters and their metadata. Fortunately since Python 3.8+, the concept of shared_memory has been introduced to share data structures between processes. It relies on posix mmap2 under the hood.

EOScale relies on this concept to store and share large satellite images and their metadatas between processes without **duplicating memory space**.

Currently, EOScale provides 2 paradigms: 
- A generic N image to M image filter that uses a tiling strategy with the concept of stability margin to parallelize local dependency algorithms while ensuring identical results. All the complexity is done for you, you just have to define your algorithm as a callable function that takes as input a list of numpy arrays, a list of their corresponding image metadata and your filter parameters as a Python dictionnary and that is all !
- A generic N image to M scalars that can returns anything that can be concatenated in a Map/Reduce paradigm. For example a histogram or statiscal values such as min, max or average.

## Your pipeline in memory !

One other great advantage of EOScale is how easy it is to chain your filters through your pipeline **in memory** and again while minimizing your memory footprint. This allows your programs to be more efficient and less consuming regarding your energy footprint. 

## Want to use it ?

Just clone this repo and pip install it ;)

The only requirement is to use a version of Python greater or equal than 3.8

## Quick start: run the example

Look at the source file pipeline.py in the directory eoscale/examples/ to see how to use EOScale.

From the root directory of eoscale, you can run:
```
python examples/pipeline.py
```

## Want to contribute ?

Here’s how it generally works:


1. Clone the project.
2. Create a topic branch from master.
3. Make some commits to improve the project.
4. Open a Merge Request when your are done.
5. The project owners will examine your features and corrections.
6. Discussion with the project owners about those commits
7. The project owners merge to master and close the merge request.

Actual project owner: [Pierre Lassalle](pierre.lassalle@cnes.fr)



