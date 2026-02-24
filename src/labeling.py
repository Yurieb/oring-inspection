import numpy as np

def connected_components(binary_img):
    rows, cols = binary_img.shape
    labels = np.zeros((rows, cols), dtype=int)

    current_label = 1

    for x in range(rows):
        for y in range(cols):

            # If pixel is white and not yet labeled
            if binary_img[x, y] == 255 and labels[x, y] == 0:

                # Start flood-fill 
                stack = [(x, y)]
                labels[x, y] = current_label

                while stack:
                    cx, cy = stack.pop()

                    # Checks all 8 neighbours
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:

                            nx = cx + dx
                            ny = cy + dy

                            # Check all 8 neighbours
                            if 0 <= nx < rows and 0 <= ny < cols:

                                if binary_img[nx, ny] == 255 and labels[nx, ny] == 0:
                                    labels[nx, ny] = current_label
                                    stack.append((nx, ny))

                current_label += 1

    return labels, current_label - 1

# Calculate area of number of pixels for each labeled component
def component_areas(labels, num_objects):
    areas = []

    for label in range(1, num_objects + 1):
        area = (labels == label).sum()
        areas.append(area)

    return areas

# Keep only the largest connected component
def keep_largest_component(labels, areas):
  
    largest_label = areas.index(max(areas)) + 1

    clean = (labels == largest_label).astype(np.uint8) * 255

    return clean

