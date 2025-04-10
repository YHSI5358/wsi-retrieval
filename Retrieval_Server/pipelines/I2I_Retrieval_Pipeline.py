import sys, cProfile, pstats
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from io import StringIO
from Retrieval_Server.modules.init_database import Initial_Database
from Retrieval_Server.modules.I2I_Retrieval.vector_retriever import Image2Image_Retriever



def main_i2i_retrieval_server_pipeline(query_img_path, vector_retriever, top_k=5):
    
    retrieved_images_nodes = vector_retriever.retrieve(query_img_path, top_k=top_k)

    return retrieved_images_nodes

def main(
        database, vector_retriever
        ):
    # struc_file = "Retrieval_Server/data/struc.sql"
    # data_file = "Retrieval_Server/data/data.sql"

    # database = Initial_Database(struc_file, data_file)
    # vector_retriever = Image2Image_Retriever(database)

    query_img_path = "Retrieval_Server/caches/patch_image/IBL015-7-15_2_2048_8192.png"       # TODO: image path
    retrieved_images_nodes = main_i2i_retrieval_server_pipeline(query_img_path, vector_retriever, top_k=10)
    
    return retrieved_images_nodes


if __name__ == "__main__":

    struc_file = "Retrieval_Server/data/struc.sql"
    data_file = "Retrieval_Server/data/data.sql"
    database = Initial_Database(struc_file, data_file)
    vector_retriever = Image2Image_Retriever(database)

    print(len(database.image_nodes))

    profile = cProfile.Profile()
    profile.enable()
    main(database, vector_retriever)
    # main()
    profile.disable()

    s = StringIO()
    ps = pstats.Stats(profile, stream=s).sort_stats('tottime')
    ps.print_stats(50)
    print(s.getvalue())

    retrieved_images_nodes = main(database, vector_retriever)
    # retrieved_images_nodes = main()
    for node in retrieved_images_nodes:
        print("Image_url", node.image_path)
        print("Level: ", node.metadata["level"])
        print("Position: ", node.metadata["position"])
        print("Size: ", node.metadata["size"])
        print("wsi_image_source: ", node.metadata["wsi_image_source"], "\n")
