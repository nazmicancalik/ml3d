"""Export to disk"""


def export_mesh_to_obj(path, vertices, faces):
    """
    exports mesh as OBJ
    :param path: output path for the OBJ file
    :param vertices: Nx3 vertices
    :param faces: Mx3 faces
    :return: None
    """

    # write vertices starting with "v "
    # write faces starting with "f "

    # ###############
    f = open(path, "a")
    f.truncate(0) # remove old content
    for vertex in vertices:
        f.write('v ' + ' '.join(map(str,vertex)) + '\n')
    for face in faces:
        f.write('f ' + ' '.join(map(str,face))+ '\n')
    f.close()
    # ###############


def export_pointcloud_to_obj(path, pointcloud):
    """
    export pointcloud as OBJ
    :param path: output path for the OBJ file
    :param pointcloud: Nx3 points
    :return: None
    """

    # ###############
    # TODO: Implement
    # ###############
