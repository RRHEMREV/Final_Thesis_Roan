# Detailed notes: `Data_Prep.py`

## Determining Opposing Pairs in `_calculate_slopes(...)`

Explanation of code that determines opposing pairs:

```
# Identify opposing pairs (see Notes.md)
opposing_pairs = []
for j, (dx, dy) in enumerate(relative_positions):
    for k, (dx_op, dy_op) in enumerate(relative_positions):
        if (
            j != k 
            and cma.np.isclose(dx, -dx_op, rtol=1.e-5, atol=1.e-8) 
            and cma.np.isclose(dy, -dy_op, rtol=1.e-5, atol=1.e-8)
            ):
            opposing_pairs.append((j, k))
```

#### Explanation with Step Size 100, Center Point at (100, 100), and 8 Neighbors:
Suppose we have a central point at `(100, 100)` and 8 neighbors at the following coordinates:
- Neighbor 0: `(100, 200)` (North)
- Neighbor 1: `(200, 200)` (Northeast)
- Neighbor 2: `(200, 100)` (East)
- Neighbor 3: `(200, 0)` (Southeast)
- Neighbor 4: `(100, 0)` (South)
- Neighbor 5: `(0, 0)` (Southwest)
- Neighbor 6: `(0, 100)` (West)
- Neighbor 7: `(0, 200)` (Northwest)

The **relative positions** of these neighbors with respect to the central point `(100, 100)` are:
- Neighbor 0: `(dx, dy) = (0, 100)` (North)
- Neighbor 1: `(dx, dy) = (100, 100)` (Northeast)
- Neighbor 2: `(dx, dy) = (100, 0)` (East)
- Neighbor 3: `(dx, dy) = (100, -100)` (Southeast)
- Neighbor 4: `(dx, dy) = (0, -100)` (South)
- Neighbor 5: `(dx, dy) = (-100, -100)` (Southwest)
- Neighbor 6: `(dx, dy) = (-100, 0)` (West)
- Neighbor 7: `(dx, dy) = (-100, 100)` (Northwest)

The code iterates through all pairs of neighbors (`j` and `k`) and checks:
1. `j != k`: Ensures the two neighbors are distinct.
2. `cma.np.isclose(dx, -dx_op)` and `cma.np.isclose(dy, -dy_op)`: Ensures the relative positions are opposite.

#### Step-by-Step Execution:
1. **Iteration 1** (`j=0`, Neighbor 0 `(100, 200)`):
   - Compare with `k=4` (Neighbor 4 `(100, 0)`):
     - `dx = 0`, `dx_op = 0` → Opposite in `dy`: `dy = 100`, `dy_op = -100`.
     - Pair `(0, 4)` is opposing.
   - Compare with other neighbors: Not opposing.

2. **Iteration 2** (`j=1`, Neighbor 1 `(200, 200)`):
   - Compare with `k=5` (Neighbor 5 `(0, 0)`):
     - `dx = 100`, `dx_op = -100` → Opposite in both `dx` and `dy`: `dy = 100`, `dy_op = -100`.
     - Pair `(1, 5)` is opposing.
   - Compare with other neighbors: Not opposing.

3. **Iteration 3** (`j=2`, Neighbor 2 `(200, 100)`):
   - Compare with `k=6` (Neighbor 6 `(0, 100)`):
     - `dx = 100`, `dx_op = -100` → Opposite in `dx`: `dy = 0`, `dy_op = 0`.
     - Pair `(2, 6)` is opposing.
   - Compare with other neighbors: Not opposing.

4. **Iteration 4** (`j=3`, Neighbor 3 `(200, 0)`):
   - Compare with `k=7` (Neighbor 7 `(0, 200)`):
     - `dx = 100`, `dx_op = -100` → Opposite in both `dx` and `dy`: `dy = -100`, `dy_op = 100`.
     - Pair `(3, 7)` is opposing.
   - Compare with other neighbors: Not opposing.

5. **Remaining Iterations**:
   - For `j=4` to `j=7`, the opposing pairs have already been identified in earlier iterations.

#### Result:
The `opposing_pairs` list will contain:
```python
opposing_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
```

#### Summary:
This code identifies pairs of neighbors that are directly opposite to each other relative to the central point `(100, 100)`. In this example:
- Neighbor 0 (North) and Neighbor 4 (South) are opposing.
- Neighbor 1 (Northeast) and Neighbor 5 (Southwest) are opposing.
- Neighbor 2 (East) and Neighbor 6 (West) are opposing.
- Neighbor 3 (Southeast) and Neighbor 7 (Northwest) are opposing.

## Processing Opposing Pairs in `_calculate_slopes(...)` (Lines 216-255 in `Data_Prep.py`)

```
# The dx dy directions and height differences for all opposing pairs
for j, k in opposing_pairs:
      neighbor_idx1 = neighbors[j]
      neighbor_idx2 = neighbors[k]

      dx = x_coords[neighbor_idx1] - x_coords[neighbor_idx2]
      dy = y_coords[neighbor_idx1] - y_coords[neighbor_idx2]
      dz = z_values[neighbor_idx1] - z_values[neighbor_idx2]

      # Saving only the positive values dz (and thus the directions in dx dy, these will be pointing 'upwards')
      if dz > 0:
         dx_list.append(float(dx))
         dy_list.append(float(dy))
         dz_list.append(float(dz))

dx_array = np.array(dx_list)
dy_array = np.array(dy_list)
dz_array = np.array(dz_list)
slope_array = dz_array/(np.sqrt(dx_array**2 + dy_array**2)) #delta_z / distance

if len(slope_array) > 0:

      # Calculate vecor magnitude of the slopes -> D = np.sqrt(A^2 + B^2 + C^2 + ...)
      slopes[i] = np.sqrt(np.sum(slope_array**2))

      # Calculate the slope direction using the weighted values of dx dy based on corresponding height difference dz
      weight = slope_array / np.sum(slope_array)
      dx_list_weighted = dx_array * weight
      dy_list_weighted = dy_array * weight

      # Transfer dx dy into degrees, increasing counterclockwise:
      # -> np.arctan2(dy=0, dx=1) = 0 degrees (East)
      # -> np.arctan2(dy=1, dx=0) = 90 degrees (North)
      # -> np.arctan2(dy=0, dx=-1) = 180 degrees (West)
      # -> np.arctan2(dy=-1, dx=0) = 270 degrees (South)
      dir_rad = np.arctan2(np.sum(dy_list_weighted), np.sum(dx_list_weighted))
      dir_degr = np.degrees(dir_rad)

      # Ensure the angle is in the range [0, 360)
      slope_dir[i] = dir_degr + 360 if dir_degr < 0 else dir_degr
```

### **Purpose**
This section of the code processes the **opposing pairs** of neighbors identified earlier in the `_calculate_slopes` function. It calculates the slope magnitudes and directions for each spatial data point based on the differences in x, y, and z coordinates between opposing pairs.

---

### **Code Breakdown**

#### **1. Loop Through Opposing Pairs**
```python
for j, k in opposing_pairs:
    neighbor_idx1 = neighbors[j]
    neighbor_idx2 = neighbors[k]
```
- **What It Does**:
  - Iterates through all opposing pairs of neighbors (`j, k`) for the current spatial data point.
  - Retrieves the indices of the two neighbors (`neighbor_idx1` and `neighbor_idx2`) from the `neighbors` array.

- **Example**:
  - Suppose `opposing_pairs = [(0, 4), (1, 5)]` and `neighbors = [2, 3, 4, 5, 6]`.
  - For the first pair `(0, 4)`:
    - `neighbor_idx1 = neighbors[0] = 2`
    - `neighbor_idx2 = neighbors[4] = 6`

---

#### **2. Calculate Differences in Coordinates**
```python
dx = x_coords[neighbor_idx1] - x_coords[neighbor_idx2]
dy = y_coords[neighbor_idx1] - y_coords[neighbor_idx2]
dz = z_values[neighbor_idx1] - z_values[neighbor_idx2]
```
- **What It Does**:
  - Computes the differences in x, y, and z coordinates between the two neighbors in the opposing pair:
    - `dx`: Difference in x-coordinates.
    - `dy`: Difference in y-coordinates.
    - `dz`: Difference in z-values (elevation).

- **Example**:
  - Suppose:
    - `x_coords = [100, 200, 300, 400, 500]`
    - `y_coords = [50, 150, 250, 350, 450]`
    - `z_values = [10, 20, 30, 40, 50]`
  - For `neighbor_idx1 = 2` and `neighbor_idx2 = 6`:
    - `dx = x_coords[2] - x_coords[6] = 300 - 500 = -200`
    - `dy = y_coords[2] - y_coords[6] = 250 - 450 = -200`
    - `dz = z_values[2] - z_values[6] = 30 - 50 = -20`

---

#### **3. Filter for Positive Elevation Differences**
```python
if dz > 0:
    dx_list.append(float(dx))
    dy_list.append(float(dy))
    dz_list.append(float(dz))
```
- **What It Does**:
  - Only considers pairs where `dz > 0` (i.e., the elevation difference is positive).
  - Appends the `dx`, `dy`, and `dz` values to their respective lists (`dx_list`, `dy_list`, `dz_list`).

- **Why?**:
  - Positive `dz` indicates that the slope is "uphill" from one neighbor to the other, which is relevant for slope calculations.

- **Example**:
  - If `dz = -20`, the pair is skipped.
  - If `dz = 10`, the values are appended:
    - `dx_list = [-200]`
    - `dy_list = [-200]`
    - `dz_list = [10]`

---

#### **4. Convert Lists to Arrays**
```python
dx_array = np.array(dx_list)
dy_array = np.array(dy_list)
dz_array = np.array(dz_list)
```
- **What It Does**:
  - Converts the lists of `dx`, `dy`, and `dz` values into numpy arrays for efficient numerical operations.

---

#### **5. Calculate Slope Magnitudes**
```python
slope_array = dz_array / (np.sqrt(dx_array**2 + dy_array**2))  # delta_z / distance
```
- **What It Does**:
  - Computes the slope for each opposing pair as the ratio of elevation difference (`dz`) to the horizontal distance (`sqrt(dx^2 + dy^2)`).

- **Example**:
  - Suppose:
    - `dx_array = [-200]`
    - `dy_array = [-200]`
    - `dz_array = [10]`
  - Then:
    - `distance = sqrt((-200)^2 + (-200)^2) = sqrt(40000 + 40000) = 282.84`
    - `slope = dz / distance = 10 / 282.84 ≈ 0.0354`

---

#### **6. Aggregate Slope Magnitudes**
```python
if len(slope_array) > 0:
    slopes[i] = np.sqrt(np.sum(slope_array**2))
```
- **What It Does**:
  - Computes the total slope magnitude for the current spatial data point as the vector magnitude of all individual slopes:
    - `slopes[i] = sqrt(A^2 + B^2 + C^2 + ...)`

- **Why?**:
  - This aggregates the contributions of all opposing pairs into a single slope magnitude.

---

#### **7. Calculate Weighted Slope Direction**
```python
weight = slope_array / np.sum(slope_array)
dx_list_weighted = dx_array * weight
dy_list_weighted = dy_array * weight
```
- **What It Does**:
  - Normalizes the slopes (`slope_array`) to create weights for each opposing pair.
  - Computes the weighted `dx` and `dy` components based on these weights.

- **Why?**:
  - Directions associated with steeper slopes have a greater influence on the final slope direction.

---

#### **8. Compute Slope Direction in Degrees**
```python
dir_rad = np.arctan2(np.sum(dy_list_weighted), np.sum(dx_list_weighted))
dir_degr = np.degrees(dir_rad)
```
- **What It Does**:
  - Calculates the overall slope direction using the weighted `dx` and `dy` components:
    - `arctan2(dy, dx)` computes the angle (in radians) of the vector formed by the weighted components.
    - `np.degrees` converts the angle from radians to degrees.

- **Example**:
  - Suppose:
    - `dx_list_weighted = [-100]`
    - `dy_list_weighted = [-100]`
  - Then:
    - `dir_rad = arctan2(-100, -100) = -2.356 radians`
    - `dir_degr = degrees(-2.356) ≈ -135 degrees`

---

#### **9. Adjust Direction to Range [0, 360)**
```python
slope_dir[i] = dir_degr + 360 if dir_degr < 0 else dir_degr
```
- **What It Does**:
  - Ensures the slope direction is expressed in the range `[0, 360)`:
    - Adds `360` to negative angles.

- **Example**:
  - If `dir_degr = -135`, the adjusted direction is:
    - `slope_dir[i] = -135 + 360 = 225 degrees`

---

### **Summary**
This code calculates the slope magnitude and direction for a spatial data point based on its opposing neighbor pairs:
1. **Slope Magnitude**:
   - Aggregates the slopes of all opposing pairs into a single value.
2. **Slope Direction**:
   - Computes a weighted average direction, ensuring that steeper slopes have a greater influence.
3. **Output**:
   - `slopes[i]`: Total slope magnitude for the point.
   - `slope_dir[i]`: Slope direction in degrees, within `[0, 360)`.

This information is crucial for understanding the terrain's steepness and orientation at each point.