# Detailed notes: `Adoption.py`

## Constraints used in: `round_0_adoption(self)`

### 1. **Self-Loop constraint**

To ensure that the current 'child-searching' parent does not pick itself as a child.

```
if parent_idx == child_idx:
    continue
```

### 2. **Both-Way constraint**

To ensure that the current 'child-searching' parent does not pick its own parent as a child. Since in round zero we dont yet have 
a complete dag object, we can't use `nx.ancestors(dag, parent_idx)`.

```
if dag.has_edge(child_idx, parent_idx):
    continue
```

## Constraints used in: `_round_i_adoption(self, dag, parent_child_dict, parent_child_dict_total)`

### 1. **Double-Round constraint**

To ensure that the current 'child-searching' parent did not allready have an adoption round.

```
if parent_idx in list(parent_child_dict.keys()):
    continue
```

### 2. **Self-Loop constraint**

To ensure that the current 'child-searching' parent does not pick itself as a child.

```
if parent_idx == child_idx:
    continue
```

### 3. **Max-Distance constraint**

To ensure that the current 'child-searching' parent does not pick a child from the other side of a domain, the maximum distance
should have a similar magnitutde as the ???? .

```
if self._is_within_max_distance(child_idx, parent_idx) == False:
    continue
```

### 4. **Ancestor constraint**

To ensure that the current 'child-searching' parent does not pick an ancestor of its own, to prevent cyclic relations within the 
DAG.

```
if child_idx in nx.ancestors(dag, parent_idx):
    continue
```

### 5. **Twin constraint**

To ensure that the current 'child-searching' parent does not pick a child that he allready has adopted before. Thus preventing
double entries (of the same child) in the values for this parent (=key) in the `parent_child_dict_i`.

```
if child_idx in parent_child_dict_i[parent_idx]:
    continue
```