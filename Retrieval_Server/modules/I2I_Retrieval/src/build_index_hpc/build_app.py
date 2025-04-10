import sys
sys.path.append(".")
sys.path.append("..")
from flask import Flask, request, jsonify
from Retrieval_Server.modules.I2I_Retrieval.src.build_index_hpc.embedding import ImagePatchEncoder

app = Flask(__name__)

image_encoder = None

def get_image_encoder():
    global image_encoder
    if image_encoder is None:
        image_encoder = ImagePatchEncoder()
    return image_encoder

@app.route('/wsi_image_embeddings', methods=['POST'])
def process_text2text_retrieval():
    wsi_name = request.json.get('wsi_name')     # str

    encoder = get_image_encoder()

    patch_infos, patch_embeddings = encoder.get_wsi_patch_info_and_emb(wsi_name)

    return jsonify({
        'patch_infos':patch_infos,
        'patch_embeddings': patch_embeddings
    })



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port="9999")