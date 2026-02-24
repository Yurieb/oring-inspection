import numpy as np

def connected_components(binary_img):
    rows, cols = binary_img.shape
    labels = np.zeros((rows, cols), dtype=int)

    current_label = 1

    for x in range(rows):
        for y in range(cols):

            if binary_img[x, y] == 255 and labels[x, y] == 0:

                stack = [(x, y)]
                labels[x, y] = current_label

                while stack:
                    cx, cy = stack.pop()

                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:

                            nx = cx + dx
                            ny = cy + dy

                            if 0 <= nx < rows and 0 <= ny < cols:

                                if binary_img[nx, ny] == 255 and labels[nx, ny] == 0:
                                    labels[nx, ny] = current_label
                                    stack.append((nx, ny))

                current_label += 1

    return labels, current_label - 1

def component_areas(labels, num_objects):
    areas = []

    for label in range(1, num_objects + 1):
        area = (labels == label).sum()
        areas.append(area)

    return areas

