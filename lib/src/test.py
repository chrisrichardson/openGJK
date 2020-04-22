import opengjkc
import opengjk
from scipy.spatial.transform import Rotation as R
import numpy as np
import pytest


def test_comparison():
    # tri = np.array([[-1, -1, 0], [1, 0, 1], [1, 0, -1]], dtype="float64")
    tet = np.array([[-1, 1, 0], [-1, -1, 0], [1, 0, 1],
                    [1, 0, -1]], dtype="float64")
    # square = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0],
    #                    [-1, 1, 0]], dtype="float64")
    cube = np.array([[1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
                     [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1]],
                    dtype="float64")

    # pt = np.array([[0, 0, 1]], dtype="float64")

    for th in np.arange(0, 2*np.pi, 0.01*np.pi):
        x = 2.2*np.cos(th)
        y = 2.2*np.sin(th)
        z = 0.0
        r = R.from_euler('zyx', [[5, 1, 2], [5, 1, 2], [
            5, 1, 2], [5, 1, 2]], degrees=True)
        tet = r.apply(tet)
        d1 = opengjkc.gjk(tet, cube + np.array([[x, y, z]]))
        d2 = opengjk.gjk(tet, cube + np.array([[x, y, z]]))
        assert(np.isclose(d1, d2, atol=1e-15))


def distance_point_to_line_3D(P1, P2, point):
    """
    distance from point to line
    """
    return np.linalg.norm(np.cross(P2-P1, P1-point))/np.linalg.norm(P2-P1)


def distance_point_to_plane_3D(P1, P2, P3, point):
    """
    Distance from point to plane
    """
    return np.abs(np.dot(np.cross(P2-P1, P3-P1) /
                         np.linalg.norm(np.cross(P2-P1, P3-P1)), point-P2))


@pytest.mark.parametrize("delta", [0.1, 1e-12, 0, -2])
def test_line_point_distance(delta):
    line = np.array([[0.1, 0.2, 0.3], [0.5, 0.8, 0.7]], dtype=np.float64)
    point_on_line = line[0] + 0.27*(line[1]-line[0])
    normal = np.cross(line[0], line[1])
    point = point_on_line + delta * normal
    distance = opengjk.gjk(line, point)
    actual_distance = distance_point_to_line_3D(
        line[0], line[1], point)
    print(distance, actual_distance)
    assert(np.isclose(distance, actual_distance))


@pytest.mark.parametrize("delta", [0.1**(3*i) for i in range(6)])
def test_tri_collision2d(delta):
    tri_1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    tri_2 = np.array([[1, delta, 0], [3, 1.2, 0], [
        1, 1, 0]], dtype=np.float64)
    P1 = tri_1[2]
    P2 = tri_1[1]
    point = tri_2[0]
    actual_distance = distance_point_to_line_3D(P1, P2, point)
    distance = opengjk.gjk(tri_1, tri_2)
    print("Computed distance ", distance, actual_distance)
    assert(np.isclose(distance, actual_distance))


@pytest.mark.parametrize("delta", [1*0.5**(3*i) for i in range(7)])
def test_tetra_distance_3d(delta):
    tetra_1 = np.array([[0, 0, 0.2], [1, 0, 0.1], [0, 1, 0.3],
                        [0, 0, 1]], dtype=np.float64)
    tetra_2 = np.array([[0, 0, -3], [1, 0, -3], [0, 1, -3],
                        [0.5, 0.3, -delta]], dtype=np.float64)
    actual_distance = distance_point_to_plane_3D(tetra_1[0], tetra_1[1],
                                                 tetra_1[2], tetra_2[3])
    distance = opengjk.gjk(tetra_1, tetra_2)
    print("Computed distance ", distance, actual_distance)
    assert(np.isclose(distance, actual_distance))


@pytest.mark.parametrize("delta", [1*0.1**(3*i) for i in range(6)])
def test_tetra_collision_3d(delta):
    tetra_1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                        [0, 0, 1]], dtype=np.float64)
    tetra_2 = np.array([[0, 0, -3], [1, 0, -3], [0, 1, -3],
                        [0.5, 0.3, -delta]], dtype=np.float64)
    actual_distance = distance_point_to_plane_3D(tetra_1[0], tetra_1[1],
                                                 tetra_1[2], tetra_2[3])
    distance = opengjk.gjk(tetra_1, tetra_2)
    print("Computed distance ", distance, actual_distance)
    assert(np.isclose(distance, actual_distance))
