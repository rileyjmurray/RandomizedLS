I'm initializing this repository with one or more basic implementations of
Blendenpik-like least-squares solvers. The solvers might use Numba acceleration.
Right now they use SciPy for trig transforms. These implementations should be
changed to use pyFFTW once pyFFTW version 0.13 is released (that version of pyFFTW
will contain functionality for the discrete cosine transform).

The protomodules folder will contain the basic implementations.
Illustrations of the basic implementations will be in the scripts folder.

Ideally, the prls package should have ristretto as a dependency.
However, ristretto's pip version hasn't been updated since August 2017.

------------------------

At end-of-day:
 * acm204_linsys.py contains code from my project for the Caltech graduate course "ACM 204".
   The code has been modified so the variable names are less tied to my use-case in optimization
   algorithms. However almost all of the utility functions are defined in terms of a wide matrix
   named "At" rather than a tall matrix named A. I added a blendenpik_srct function that
   nevertheless accepts input in terms of a tall matrix A rather than a wide matrix At.
 * test_acm204_linsys.py contains unittests for the main functions implemented in acm204_linsys.py.
   This file does not contain any unittests for "blendenpik_srct"!.
   
If anyone intends to run these scripts, you should make sure your PYTHONPATH environment
variable includes the root directory of this git repository. If you use an IDE like PyCharm
you shouldn't need to set that environment variable manually.
