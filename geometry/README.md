The STL geometry follows a standard NACA-0012 airfoil outline (in the x-y-plane). For meshing in OpenFOAM, it has a depth in z of 0.2 (it extends from -0.1 to 0.1). The trailing edge is not sharp but blunt.

<img src="naca0012.png" alt="naca0012">

# The scaled file is derived from naca0012.stl by :
```
$ surfaceTransformPoints -scale '(1.5 3.5 1)' -translate '(-0.1 0 0)' naca0012.stl scaled_airfoil.stl '
```
 -> which scales naca0012 by 1.5 times in x direction and 3.5 times in y direction. It also translates naca0012 by 0.1 in negative x direction.