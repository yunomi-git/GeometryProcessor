import numpy as np

def distance_point_from_axis(origin, axis, point):
    projections = np.empty(len(point))
    for i in range(len(point)):
        projections[i] = np.dot(point[i] - origin, axis)
    projection_onto_axis = np.outer(projections, axis) + origin
    # projection_onto_axis = np.dot(point - origin, axis) * axis + origin
    distance_from_projection = np.linalg.norm(point - projection_onto_axis, axis=1)
    return distance_from_projection


def point_onto_axis_distance_along_axis(origin, axis, point):
    projections = np.empty(len(point))
    for i in range(len(point)):
        projections[i] = np.abs(np.dot(point[i] - origin, axis))
    return projections
    # return np.dot(point - origin, axis)


