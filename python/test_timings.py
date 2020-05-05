import numpy as np
import opengjkc as opengjk
from scipy.spatial.transform import Rotation as R
import time


def test():
    cubes = [np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                       [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]],
                      dtype=np.float64)]

    r = R.from_euler('z', 45, degrees=True)
    cubes.append(r.apply(cubes[0]))
    r = R.from_euler('y', np.arctan2(1.0, np.sqrt(2)))
    cubes.append(r.apply(cubes[1]))
    r = R.from_euler('y', 45, degrees=True)
    cubes.append(r.apply(cubes[0]))


    tmin = 1e6
    tmax = 0.0
    t0 = 0.0
    for j in range(1000):
        for c0 in range(4):
            for c1 in range(4):
                dx = cubes[c0][:,0].max() - cubes[c1][:,0].min()
                cube0 = cubes[c0]

                for delta in [1.0, 1e-2, 1e-4, 1e-8]:
                    cube1 = cubes[c1] + np.array([dx + delta, 0, 0])
                    distance = opengjk.gjk_timed(cube0, cube1)
                    tmin = min(tmin, distance[0])
                    tmax = max(tmax, distance[0])
                    t0 += distance[0]
                    # assert np.isclose(distance, delta), "Error with " + str(c0) + str(c1) + " " + str(distance) + " " + str(delta)

    print('Time mean = ',t0/64000 * 1e6 , " us")
    print('Time best = ',tmin * 1e6 , " us")
    print('Time worst = ',tmax * 1e6 , " us")

test()
