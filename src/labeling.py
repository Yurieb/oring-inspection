import numpy as np

# Labels every separate white region in the image with a unique number
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

                    # Check all 8 neighbours
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


# Calculate area number of pixels for each labeled component
def component_areas(labels, num_objects):
    areas = []

    for label in range(1, num_objects + 1):
        area = (labels == label).sum()
        areas.append(area)

    return areas


# Calculate perimeter by counting boundary pixels of a component
def compute_perimeter(clean_component):
    rows, cols = clean_component.shape
    perimeter = 0

    for x in range(1, rows - 1):
        for y in range(1, cols - 1):

            # If pixel is white check if any neighbour is black boundary pixel
            if clean_component[x, y] == 255:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if clean_component[x + dx, y + dy] == 0:
                            perimeter += 1
                            break
                    else:
                        continue
                    break

    return perimeter


# Circularity — perfect circle = 1.0  broken or irregular ring will score lower
def compute_circularity(area, perimeter):
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


# Return only the largest white region this will be the ring
def keep_largest_component(labels, areas):
    largest_label = areas.index(max(areas)) + 1
    clean = (labels == largest_label).astype(np.uint8) * 255
    return clean

# Measure the hole inside the ring
# Flood fills inward from the image border through black pixels
# Any black pixels the fill cannot reach are inside the ring
def find_interior_hole_area(binary_ring):
    rows, cols = binary_ring.shape
    reachable = np.zeros((rows, cols), dtype=bool)
    stack = []

    # Start the fill from every black pixel on the image border
    for x in range(rows):
        for y in range(cols):
            if x == 0 or x == rows - 1 or y == 0 or y == cols - 1:
                if binary_ring[x, y] == 0 and not reachable[x, y]:
                    reachable[x, y] = True
                    stack.append((x, y))

    # Spread through connected black pixels
    while stack:
        cx, cy = stack.pop()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if binary_ring[nx, ny] == 0 and not reachable[nx, ny]:
                    reachable[nx, ny] = True
                    stack.append((nx, ny))

    # Black pixels the fill never reached are the interior hole
    hole_area = int((~reachable & (binary_ring == 0)).sum())
    return hole_area