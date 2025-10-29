from math import sin, cos, pi
from neo4j import GraphDatabase

# ====== 配置这里改成你的 ======
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "neuroxiv"
# ===========================

# 布局半径
R_SUB   = 200.0  # 中圈半径: subregion
R_CHILD = 300.0  # 外圈半径: subclass
CHILD_ANGLE_OFFSET = 0.08  # 两个subclass在父角度左右各偏移约0.08弧度 (~4.5°)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# 第1步: 拉取CLA -> top20 subregions -> 每个sub的两个代表subclass
cypher_extract = """
MATCH (cla:Region {acronym:'CLA'})-[rl:PROJECT_TO]->(sub:Subregion)
WITH cla, sub, rl, rl.weight AS projection_weight
ORDER BY projection_weight DESC
LIMIT 20
WITH cla, collect({sub: sub, weight: projection_weight}) AS subs

UNWIND range(0, size(subs)-1) AS idx
WITH cla,
     subs[idx].sub     AS sub,
     subs[idx].weight  AS weight,
     idx               AS sub_rank,
     size(subs)        AS sub_total

CALL {
    WITH sub
    MATCH (sub)-[:HAS_SUBCLASS]->(child)
    WITH child
    ORDER BY child.acronym ASC
    LIMIT 2
    RETURN collect(child) AS kids
}
RETURN
    id(cla)        AS cla_id,
    id(sub)        AS sub_id,
    weight         AS sub_weight,
    sub_rank       AS sub_rank,
    sub_total      AS sub_total,
    [kid IN kids | id(kid)] AS subclass_ids;
"""

with driver.session() as session:
    rows = list(session.run(cypher_extract))

# rows 是每个 subregion 一行
# 我们现在要构建一个坐标表 coords_to_set:
# { node_id -> {viz_x, viz_y, viz_layer} }

coords_to_set = {}

# 先放 CLA 在中心 (0,0)
if rows:
    cla_node_id = rows[0]["cla_id"]
    coords_to_set[cla_node_id] = {
        "viz_x": 0.0,
        "viz_y": 0.0,
        "viz_layer": 0,
    }

# 现在给每个 subregion 分配角度
# 角度theta = 2*pi * (sub_rank / sub_total)
for record in rows:
    sub_id      = record["sub_id"]
    sub_rank    = record["sub_rank"]
    sub_total   = record["sub_total"]
    subclass_ids = record["subclass_ids"]

    theta = 2.0 * pi * (float(sub_rank) / float(sub_total))

    sub_x = R_SUB * cos(theta)
    sub_y = R_SUB * sin(theta)

    coords_to_set[sub_id] = {
        "viz_x": sub_x,
        "viz_y": sub_y,
        "viz_layer": 1,
    }

    # subclass 两个，沿着父角度 ± 偏移
    # 如果只有1个，就只放在中心角度
    for i, kid_id in enumerate(subclass_ids):
        if len(subclass_ids) == 1:
            kid_theta = theta
        elif len(subclass_ids) == 2:
            # i=0 -> theta - offset/2, i=1 -> theta + offset/2
            # 这样两个孩子微微分开，像两片叶子
            if i == 0:
                kid_theta = theta - CHILD_ANGLE_OFFSET/2.0
            else:
                kid_theta = theta + CHILD_ANGLE_OFFSET/2.0
        else:
            # 理论上不会 >2，但以防万一，均匀撒开
            kid_theta = theta + (i - (len(subclass_ids)-1)/2.0) * CHILD_ANGLE_OFFSET

        kid_x = R_CHILD * cos(kid_theta)
        kid_y = R_CHILD * sin(kid_theta)

        coords_to_set[kid_id] = {
            "viz_x": kid_x,
            "viz_y": kid_y,
            "viz_layer": 2,
        }

# 把 coords_to_set 变成列表，方便 UNWIND
update_payload = [
    {
        "nid": node_id,
        "viz_x": data["viz_x"],
        "viz_y": data["viz_y"],
        "viz_layer": data["viz_layer"],
    }
    for node_id, data in coords_to_set.items()
]

# 第2步: 批量写回这些坐标到图数据库的节点属性
cypher_update = """
UNWIND $batch AS row
MATCH (n)
WHERE id(n) = row.nid
SET n.viz_x = row.viz_x,
    n.viz_y = row.viz_y,
    n.viz_layer = row.viz_layer
RETURN count(n) AS updated_nodes;
"""

with driver.session() as session:
    res = session.run(cypher_update, batch=update_payload)
    print(list(res))