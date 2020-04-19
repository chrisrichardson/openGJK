import opengjkc
import opengjk
import numpy as np

tri = np.array([[-1,-1,0],[1,0,1],[1,0,-1]], dtype="float64")
square = np.array([[1,1,0],[1,-1,0],[-1,-1,0],[-1,1,0]], dtype="float64")
cube = np.array([[1,1,1],[1,-1,1],[-1,-1,1],[-1,1,1],
                 [1,1,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1]], dtype="float64")


pt = np.array([[0,0,1]], dtype="float64")

for th in np.arange(0, 2*np.pi, 0.01*np.pi):
    x = np.sqrt(2)*np.cos(th)
    y = np.sqrt(2)*np.sin(th)
    z = 0.0
    d1 = opengjkc.gjk(tri, [[x,y,z]])
    d2 = opengjk.gjk(tri, [[x,y,z]])
    print(x, y, z, d1, d2)

