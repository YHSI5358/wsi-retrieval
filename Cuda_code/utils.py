import requests
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
import cudf
import cugraph
from itertools import combinations

def split_img_fun(query_image, other_info):
    # 以无重叠的方式切割图片为多个正方形
    width, height = query_image.size
    # 以224为一个正方形的边长, 以正方形的左上角为起点，切割图片, 保证正方形不超出图片边界
    squares = []
    for x in range(0, width, 224):
        for y in range(0, height, 224):
            if x + 224 <= width and y + 224 <= height:
                square = query_image.crop((x, y, x + 224, y + 224))
                squares.append(square)
                
    return squares

def request_image(query_img_path):
    if "http" in query_img_path:
        response = requests.get(query_img_path, verify=False)
        query_image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        query_image = Image.open(query_img_path).convert("RGB")
    return query_image

def judge_if_connected(info1, info2):
    # 判断两个region是否相邻,先判断name是否相同，再判断level是否相同，再判断左上角坐标的xy加上或者减去wh是否与另一个region的xy相同
    if info1["name"] != info2["name"]:
        return False
    if info1["level"] != info2["level"]:
        return False
    if int(info1["x"]) + int(info1["w"]) == int(info2["x"]) and int(info1["y"]) == int(info2["y"]):
        return True
    if int(info1["x"]) - int(info1["w"]) == int(info2["x"]) and int(info1["y"]) == int(info2["y"]):
        return True
    if int(info1["y"]) + int(info1["h"]) == int(info2["y"]) and int(info1["x"]) == int(info2["x"]):
        return True
    if int(info1["y"]) - int(info1["h"]) == int(info2["y"]) and int(info1["x"]) == int(info2["x"]):
        return True
    if int(info1["x"]) + int(info1["w"]) == int(info2["x"]) and int(info1["y"]) + int(info1["h"]) == int(info2["y"]):
        return True
    if int(info1["x"]) - int(info1["w"]) == int(info2["x"]) and int(info1["y"]) - int(info1["h"]) == int(info2["y"]):
        return True
    if int(info1["x"]) + int(info1["w"]) == int(info2["x"]) and int(info1["y"]) - int(info1["h"]) == int(info2["y"]):
        return True
    if int(info1["x"]) - int(info1["w"]) == int(info2["x"]) and int(info1["y"]) + int(info1["h"]) == int(info2["y"]):
        return True
    return False


def concat_url(name,x,y,w,h,level):
    return f" metaservice/api/region/openslide/{name}/{x}/{y}/{w}/{h}/{level}"


def get_url_list_cpu(combined_regions):
    url_list = []
    for combined_region in combined_regions[:]:

        # 应该先按照name分组，再按照level分组，最后对每个小分组，都返回
        x = min([int(v['x']) for v in combined_region])
        y = min([int(v['y']) for v in combined_region])
        w = max([int(v['x']) + int(v['w']) for v in combined_region]) - x
        h = max([int(v['y']) + int(v['h']) for v in combined_region]) - y
        # level为多数投票后得到的level
        level2num = {}
        for v in combined_region:
            level = v['level']
            if level in level2num:
                level2num[level] += 1
            else:
                level2num[level] = 1
        level = max(level2num, key=level2num.get)
        id = combined_region[0]['id']
        name = combined_region[0]['name']
        name = str(id)+ "_" + name
        url = concat_url(name,x,y,w,h,level)
        url_list.append(url)
    return url_list

def get_url_list(combined_regions, group_sizes):
    # 获取group_sizes的index    
    group_sizes_index = group_sizes.index.to_pandas()
    # 将 cuDF DataFrame 转换为 pandas DataFrame
    combined_regions_pd = combined_regions.to_pandas()
    url_list = []
    for label in group_sizes_index[:]:
        vertex = combined_regions_pd[combined_regions_pd['labels'] == label]['vertex']
        # 应该先按照name分组，再按照level分组，最后对每个小分组，都返回
        x = min([int(v['x']) for v in vertex])
        y = min([int(v['y']) for v in vertex])
        w = max([int(v['x']) + int(v['w']) for v in vertex]) - x
        h = max([int(v['y']) + int(v['h']) for v in vertex]) - y
        # level为多数投票后得到的level
        level2num = {}
        for v in vertex:
            level = v['level']
            if level in level2num:
                level2num[level] += 1
            else:
                level2num[level] = 1
        level = max(level2num, key=level2num.get)
        id = vertex.iloc[0]['id']
        name = vertex.iloc[0]['name']
        name = str(id)+ "_" + name
        url = concat_url(name,x,y,w,h,level)
        url_list.append(url)
    return url_list

def get_result_embeddings(url_list, image_encoder):
    embeddings_list = []
    for url in url_list:
        query_image = request_image(url)
        # resize到224*224
        query_image = query_image.resize((224, 224))
        embedding = image_encoder.encode_image(query_image)
        embeddings_list.append(embedding)
    return embeddings_list

def get_combined_regions_gpu(result_infos):
    search_info_list = []
    for info in result_infos:
        search_info = {}
        level = info.split("_")[-1].split(".")[0]
        w = info.split("_")[-3]
        h = info.split("_")[-2]
        x = info.split("_")[-5]
        y = info.split("_")[-4]
        id = info.split("_")[0]
        name = "_".join(info.split("_")[1:-5])
        search_info = {"id": id, "name": name, "x": x, "y": y, "w": w, "h": h, "level": level}
        search_info_list.append(search_info)

    # 构建边列表
    edges = []
    for a, b in combinations(search_info_list, 2):  # 只考虑每对节点一次，避免重复
        if judge_if_connected(a, b):
            edges.append((a, b))

    # 将边列表转换成cudf DataFrame
    edges_df = cudf.DataFrame(edges, columns=['source', 'destination'])

    # 创建一个无向图（因为对于WCC来说，边的方向并不重要）
    G = cugraph.Graph(directed=False)

    # 添加边到图中
    G.from_cudf_edgelist(edges_df, source='source', destination='destination')

    # 计算弱连通分量
    wcc_components = cugraph.weakly_connected_components(G)
    
    # 计算弱连通分量
    # wcc_components = cugraph.wcc(G)

    return wcc_components



def get_combined_regions_cpu(result_infos):

    search_info_list = []
    
    for info in result_infos:
        search_info = {}
        level = info.split("_")[-1].split(".")[0]
        w = info.split("_")[-3]
        h = info.split("_")[-2]
        x = info.split("_")[-5]
        y = info.split("_")[-4]
        id = info.split("_")[0]
        name = "_".join(info.split("_")[1:-5])
        search_info = {"id": id, "name": name, "x": x, "y": y, "w": w, "h": h, "level": level}
        search_info_list.append(search_info)

    def dfs(node, component, visited):
        # 将当前节点标记为已访问
        visited[node] = True
        # 将当前节点添加到当前组件
        component.append(search_info_list[node])
        
        # 遍历其他所有节点
        for neighbor in range(len(search_info_list)):
            if not visited[neighbor] and judge_if_connected(search_info_list[node], search_info_list[neighbor]):
                # 如果邻居节点未被访问过且与当前节点相连，则递归地进行DFS
                dfs(neighbor, component, visited)

    # 初始化变量
    visited = [False] * len(search_info_list)
    components = []

    # 对每个节点执行DFS，如果该节点尚未被访问
    for i in range(len(search_info_list)):
        if not visited[i]:
            # 创建一个新的组件列表
            current_component = []
            # 执行DFS并填充当前组件
            dfs(i, current_component, visited)
            # 将当前组件添加到结果列表
            components.append(current_component)

    # 剔除掉只有一个region的component
    components = [component for component in components if len(component) > 1]

    return components