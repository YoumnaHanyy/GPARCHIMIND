from collections import defaultdict, deque

def compute_metrics(adl):
    services = adl.get("services", [])
    relationships = adl.get("relationships", [])

    service_names = [s["name"] for s in services]

    # ---------- Basic counts ----------
    num_components = len(services)
    num_relationships = len(relationships)

    # ---------- Fan-in / Fan-out ----------
    fan_in = defaultdict(int)
    fan_out = defaultdict(int)

    for r in relationships:
        fan_out[r["source"]] += 1
        fan_in[r["target"]] += 1

    avg_fan_out = (
        sum(fan_out.values()) / len(service_names)
        if service_names else 0
    )
    max_fan_out = max(fan_out.values(), default=0)

    # ---------- Async ratio ----------
    async_count = sum(
        1 for r in relationships if r.get("type") == "event-flow"
    )
    async_ratio = (
        async_count / num_relationships
        if num_relationships > 0 else 0
    )

    # ---------- Critical path length ----------
    graph = defaultdict(list)
    indegree = defaultdict(int)

    for r in relationships:
        src = r["source"]
        dst = r["target"]
        graph[src].append(dst)
        indegree[dst] += 1

    queue = deque()
    distance = defaultdict(int)

    for s in service_names:
        if indegree[s] == 0:
            queue.append(s)
            distance[s] = 1

    while queue:
        node = queue.popleft()
        for nbr in graph[node]:
            indegree[nbr] -= 1
            distance[nbr] = max(distance[nbr], distance[node] + 1)
            if indegree[nbr] == 0:
                queue.append(nbr)

    critical_path_length = max(distance.values(), default=0)

    return {
        "num_components": num_components,
        "num_relationships": num_relationships,
        "avg_fan_out": round(avg_fan_out, 2),
        "max_fan_out": max_fan_out,
        "async_ratio": round(async_ratio, 2),
        "critical_path_length": critical_path_length
    }
