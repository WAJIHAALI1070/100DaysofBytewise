#Graph Traversal (BFS and DFS)
from collections import deque


def bfs(graph, start):
    visited = set()
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            result.append(node)
            queue.extend(graph[node])

    return result


def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    result = [start]

    for neighbor in graph[start]:
        if neighbor not in visited:
            result.extend(dfs(graph, neighbor, visited))

    return result


# Example usage
graph = {0: [1, 2], 1: [2], 2: [0, 3], 3: [3]}
bfs_result = bfs(graph, 2)
dfs_result = dfs(graph, 2)
print("BFS result:", bfs_result)
print("DFS result:", dfs_result)
