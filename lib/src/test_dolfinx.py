import opengjk
import numpy as np
from mpi4py import MPI
import dolfinx
import dolfinx.geometry


def test_quad_dofs():
    # Test where we check if the dof coordinates of a quadrilateral cell is
    # within machine precision distance from any cell.
    # Dolfin-X master fails to identify several cells, where openGJK gets all
    # of the correctly.

    # Create mesh, function space and extract the dof coordinates
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 2, 2,
                                  dolfinx.cpp.mesh.CellType.quadrilateral)
    V = dolfinx.FunctionSpace(mesh, ("CG", 1))
    x = V.tabulate_dof_coordinates()

    # Compute map from cell vertices (topology) to cell nodes (geometry)
    c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
    vertex_to_node = np.zeros(mesh.topology.index_map(0).size_local,
                              dtype=np.int64)
    x_dofmap = mesh.geometry.dofmap
    for c in range(c_to_v.num_nodes):
        vertices = c_to_v.links(c)
        x_dofs = x_dofmap.links(c)
        for i in range(len(vertices)):
            vertex_to_node[vertices[i]] = x_dofs[i]


    points = mesh.geometry.x
    bbtree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    # Loop through all dof coordinates, first finding all possible
    # collisions with the cells boundingboxtree, then using openGJK
    # to compute the distance between the point and the cell.
    for xi in x:
        possible_cells = dolfinx.geometry.compute_collisions_point(bbtree,
                                                                   xi)[0]
        actual_cells = []
        for cell in possible_cells:
            vertices = c_to_v.links(cell)
            nodes = vertex_to_node[vertices]
            physical_points = points[nodes]
            distance = opengjk.gjk(physical_points, xi)
            # If distance close to machine precision.
            if distance < 1e-15:
                actual_cells.append(cell)

        # Compute actual collisions in the old way
        actual_cells_old = dolfinx.geometry.compute_entity_collisions_mesh(
                   bbtree, mesh, xi)[0]
        # Compare results
        print("Point", xi, "old", actual_cells_old, "new", actual_cells,
              "Distance open gjk ", distance)
        assert(len(actual_cells) > 0)
