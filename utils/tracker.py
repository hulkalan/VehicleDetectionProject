import numpy as np
from scipy.spatial import distance as dist


class VehicleTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.vehicle_count = 0
        self.counted_vehicles = set()

    def register(self, centroid):
        """Register a new object with the next available ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids, counting_line_y=None):
        """Update tracked objects with new centroids"""
        if len(input_centroids) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())

            # Compute distance matrix
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Find minimum distances and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            # Track which indices we've used
            used_row_indices = set()
            used_col_indices = set()

            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                # Update object position
                object_id = object_ids[row]
                old_centroid = self.objects[object_id]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                # Count vehicles crossing the line
                if counting_line_y is not None:
                    if (old_centroid[1] < counting_line_y and
                            input_centroids[col][1] >= counting_line_y and
                            object_id not in self.counted_vehicles):
                        self.vehicle_count += 1
                        self.counted_vehicles.add(object_id)

                used_row_indices.add(row)
                used_col_indices.add(col)

            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col])

        return self.objects
