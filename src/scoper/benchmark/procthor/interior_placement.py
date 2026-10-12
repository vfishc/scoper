import math

def interior_positions(bounds, obj):
    box = obj['axisAlignedBoundingBox']
    margin = {k: (bounds['size'][k] - box['size'][k]) / 2 - 0.002 for k in 'xyz'}
    if min(margin.values()) < 0:
        return []
    center = {k: bounds['center'][k] - (box['center'][k] - obj['position'][k]) for k in 'xyz'}
    layers = min(81, int(math.floor(2 * margin['y'] / 0.01)) + 1)
    points = []
    for level in range(layers):
        for x, z in ((0, 0), (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)):
            points.append({'x': center['x'] + x * margin['x'], 'y': center['y'] - margin['y'] + level * 0.01, 'z': center['z'] + z * margin['z']})
    return points
