import opengjkc
import opengjk
from scipy.spatial.transform import Rotation as R
import numpy as np

tri = np.array([[-1, -1, 0], [1, 0, 1], [1, 0, -1]], dtype="float64")
tet = np.array([[-1, 1, 0], [-1, -1, 0], [1, 0, 1],
                [1, 0, -1]], dtype="float64")
square = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0],
                   [-1, 1, 0]], dtype="float64")
cube = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
                 [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]], dtype="float64")


pt = np.array([[0, 0, 1]], dtype="float64")

for th in np.arange(0, 2*np.pi, 0.01*np.pi):
    x = 2.2*np.cos(th)
    y = 2.2*np.sin(th)
    z = 0.0
    r = R.from_euler('zyx', [[5, 1, 2],[5, 1, 2],[5, 1, 2], [5,1,2]], degrees=True)
    tet = r.apply(tet)
    d1 = opengjkc.gjk(tet, cube + np.array([[x, y, z]]))
    d2 = opengjk.gjk(tet, cube + np.array([[x, y, z]]))
    print(x, y, z, d1 - d2)
    assert(np.isclose(d1, d2, atol=1e-15))
