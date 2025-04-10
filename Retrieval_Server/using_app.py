import requests


# def text2text_retireval_response(query):
#     # url = "  /rag/text2text_retrieval"  # 对外
#     url = "http://10.90.156.242:5000/rag/text2text_retrieval"  # 对内
#     data = {
#         'origin_query':query,
#         'llm':'llama3'      # 可替换：“mistral”，“llama3”， “gpt4o”
#         }
#     response = requests.post(url, json=data)

#     answer = response.json()['answer']
#     metadata = response.json()['metadata']

#     return answer, metadata

# def image2image_retireval_response(query):
#    
#     # url = "  /rag/image2image_retrieval"  # 
#     url = "http://10.90.156.242:5000/rag/image2image_retrieval" # 
#     data = {
#         'query_img_path':query,
#         'top_k':10,
#         }
#     response = requests.post(url, json=data)

#     answer = response.json()['answer']          
#     retrieved_images_information = response.json()['retrieved_images_information']

#     wsi_level = [image_info["wsi_level"] for image_info in retrieved_images_information], 
#     metadata_position = [image_info["metadata_position"] for image_info in retrieved_images_information], 
#     patch_size = [image_info["patch_size"] for image_info in retrieved_images_information], 
#     source_wsi_name = [image_info["source_wsi_name"] for image_info in retrieved_images_information], 

#     return answer, wsi_level, metadata_position, patch_size, source_wsi_name

def image2image_retireval_response(query):
    "新版图像检索（大规模）"
    # url = 
    data = {
        'query_img_path':query,
        'top_k':20,
        }
    response = requests.post(url, json=data)

    answer = response.json()['answer']        
    retrieved_images_information = response.json()['retrieved_images_information']

    wsi_level = [image_info["wsi_level"] for image_info in retrieved_images_information], 
    metadata_position = [image_info["metadata_position"] for image_info in retrieved_images_information], 
    patch_size = [image_info["patch_size"] for image_info in retrieved_images_information], 
    source_wsi_name = [image_info["source_wsi_name"] for image_info in retrieved_images_information], 

    return answer, wsi_level, metadata_position, patch_size, source_wsi_name

# def text2image_retireval_response(query):
#     # url = "  /rag/text2image_retrieval"  
#     url = 
#     data = {
#         'origin_query':query,
#         'llm':'llama3'                                   
#         }
#     response = requests.post(url, json=data)

#     answer = response.json()['answer']                    
#     image_paths = response.json()['image_paths']

#     return answer, image_paths



if __name__ == "__main__":


    # answer, metadata = text2text_retireval_response(query_text)
    # print(f"Answer: {answer}")
    # print(f"Meta Content: {metadata["vector"]}")

    # query_img_path = "  /images/data-manager/1719375153-105291.png"
    query_img_path = " metaservice/api/region/openslide/282540-22.tiff/6144/6912/256/256/2"
    answer, wsi_level, metadata_position, patch_size, source_wsi_name = image2image_retireval_response(query_img_path)
    print(f"Answer: {answer}")
    print(f"wsi_level: {wsi_level}")
    print(f"metadata_position: {metadata_position}")
    print(f"patch_size: {patch_size}")
    print(f"source_wsi_name: {source_wsi_name}")

    for i in range(len(wsi_level[0])):
        retrieve_img_path = f""" metaservice/api/region/openslide/{source_wsi_name[0][i]}/{metadata_position[0][i][0]}/{metadata_position[0][i][1]}/{patch_size[0][i][0]}/{patch_size[0][i][1]}/{wsi_level[0][i]} """
        print(retrieve_img_path)



    # answer, image_paths = text2image_retireval_response(query_text)
    # print(f"Answer: {answer}")
    # print(f"Image Paths: {image_paths}")

