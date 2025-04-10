import requests
from flask import Flask, request, jsonify


app = Flask(__name__)

def image2image_retireval_response(query, top_k=20):

    url = ""
    data = {
        'query_img_path':query,
        'top_k':top_k,
        }
    response = requests.post(url, json=data)

    answer = response.json()['answer']   
    retrieved_images_result = response.json()['retrieved_images_information']

    return answer, retrieved_images_result

    wsi_level = [image_info["wsi_level"] for image_info in retrieved_images_information], 
    metadata_position = [image_info["metadata_position"] for image_info in retrieved_images_information], 
    patch_size = [image_info["patch_size"] for image_info in retrieved_images_information], 
    source_wsi_name = [image_info["source_wsi_name"] for image_info in retrieved_images_information], 

    return answer, wsi_level, metadata_position, patch_size, source_wsi_name

@app.route('/query/image2image_retrieval', methods=['POST'])
def process_image2image_retrieval():
    query_img_path = request.json.get('query_img_path')
    top_k = request.json.get('top_k')

    _, retrieved_images_result = image2image_retireval_response(query_img_path, top_k)

    retrieved_images_information = [{
            "wsi_level":result['wsi_level'],
            "metadata_position":result['metadata_position'],
            "patch_size":result['patch_size'],
            "source_wsi_name":result['source_wsi_name'],
        } for result in retrieved_images_result]

    return jsonify({  
        'answer':"Image2Image Retrieval",  
        'retrieved_images_information':retrieved_images_information
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port="7310", use_reloader=False)