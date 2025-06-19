import os
import random
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt

# Load images and convert them to feature vectors
def load_images(image_folder):
    images = []
    filenames = sorted(os.listdir(image_folder))  # Ensure sorted order
    for file in filenames:
        img_path = os.path.join(image_folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = cv2.resize(img, (100, 100)).flatten()  # Resize and flatten
        images.append(img)
    return np.array(images), filenames

# Reduce dimensionality with PCA
def reduce_dimensionality(image_vectors, n_components=50):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(image_vectors), pca

def find_closest_stage(input_image,user_img_vector, stored_vectors, stage_distribution):
    similarities = cosine_similarity(user_img_vector.reshape(1, -1), stored_vectors)[0]
    closest_img_index = np.argmax(similarities)  # Get the most similar image index

    # Convert closest index to filename (since images are stored as Img1.png, Img2.png, ...)
    closest_img_name = f"Img{closest_img_index + 1}.png"  # Adding +1 to align with filename numbering
    print(closest_img_name)
    # Find which stage this closest image belongs to
    for stage in range(len(stage_distribution)):
        for image in stage_distribution[stage]:
            if image in input_image:
                return stage
    return -1  # Return None if no match found

# New dictionary mapping stages to plant info, fertilizer recommendation options, and soil moisture options
stage_info = {
    1: {
        "plant_info": "Stage 1: Seedling stage - The plant is establishing roots and initial growth.",
        "fertilizer_recommendation": [
            "Use Nitrogen-rich fertilizer such as Urea. Amount: 50 kg/ha.",
            "Apply organic compost rich in nitrogen. Amount: 40 kg/ha.",
            "Use manure to enrich soil nitrogen. Amount: 45 kg/ha.",
            "Apply ammonium nitrate fertilizer. Amount: 55 kg/ha.",
            "Use calcium nitrate fertilizer. Amount: 50 kg/ha.",
            "Apply blood meal fertilizer. Amount: 45 kg/ha.",
            "Use feather meal fertilizer. Amount: 40 kg/ha.",
            "Apply fish emulsion fertilizer. Amount: 50 kg/ha.",
            "Use soybean meal fertilizer. Amount: 45 kg/ha.",
            "Apply cottonseed meal fertilizer. Amount: 50 kg/ha.",
            "Use bone meal fertilizer. Amount: 48 kg/ha.",
            "Apply kelp meal fertilizer. Amount: 42 kg/ha.",
            "Use green sand fertilizer. Amount: 44 kg/ha.",
            "Apply rock phosphate fertilizer. Amount: 46 kg/ha.",
            "Use composted manure fertilizer. Amount: 43 kg/ha.",
            "Apply worm castings fertilizer. Amount: 41 kg/ha.",
            "Use alfalfa meal fertilizer. Amount: 47 kg/ha.",
            "Apply soybean meal fertilizer. Amount: 49 kg/ha.",
            "Use cottonseed meal fertilizer. Amount: 45 kg/ha.",
            "Apply fish bone meal fertilizer. Amount: 50 kg/ha."
        ],
        "soil_moisture": [
            "Maintain soil moisture at 60-70% during this stage.",
            "Keep soil moist but not waterlogged, around 65%.",
            "Ensure regular watering to maintain 60% soil moisture.",
            "Maintain soil moisture at 55-65%.",
            "Keep soil moisture between 60-70%.",
            "Water moderately to keep soil moist.",
            "Avoid waterlogging, maintain moisture around 60%.",
            "Ensure soil moisture is consistent at 65%.",
            "Maintain soil moisture at 60-68%.",
            "Keep soil moisture optimal between 60-70%.",
            "Maintain soil moisture at 62-72%.",
            "Keep soil moisture steady at 64%.",
            "Water regularly to maintain 63% moisture.",
            "Avoid dry spells, keep soil moist at 61%.",
            "Ensure soil moisture is balanced at 66%.",
            "Maintain soil moisture at 59-69%.",
            "Keep soil moisture consistent at 67%.",
            "Water moderately to maintain 65% moisture.",
            "Avoid water stress, keep soil moist at 68%.",
            "Maintain soil moisture at 60-70% with regular watering."
        ]
    },
    2: {
        "plant_info": "Stage 2: Vegetative stage - Rapid growth of leaves and stems.",
        "fertilizer_recommendation": [
            "Apply balanced NPK fertilizer. Amount: 75 kg/ha.",
            "Use slow-release fertilizer with balanced nutrients. Amount: 70 kg/ha.",
            "Supplement with foliar feeding of micronutrients.",
            "Apply ammonium sulfate fertilizer. Amount: 65 kg/ha.",
            "Use magnesium sulfate fertilizer. Amount: 60 kg/ha.",
            "Apply calcium nitrate fertilizer. Amount: 70 kg/ha.",
            "Use potassium nitrate fertilizer. Amount: 68 kg/ha.",
            "Apply zinc sulfate fertilizer. Amount: 55 kg/ha.",
            "Use manganese sulfate fertilizer. Amount: 50 kg/ha.",
            "Apply copper sulfate fertilizer. Amount: 45 kg/ha.",
            "Use iron sulfate fertilizer. Amount: 48 kg/ha.",
            "Apply borax fertilizer. Amount: 40 kg/ha.",
            "Use cobalt sulfate fertilizer. Amount: 42 kg/ha.",
            "Apply molybdenum fertilizer. Amount: 38 kg/ha.",
            "Use seaweed extract fertilizer. Amount: 60 kg/ha.",
            "Apply humic acid fertilizer. Amount: 55 kg/ha.",
            "Use kelp extract fertilizer. Amount: 50 kg/ha.",
            "Apply fish emulsion fertilizer. Amount: 65 kg/ha.",
            "Use compost tea fertilizer. Amount: 58 kg/ha.",
            "Apply worm castings fertilizer. Amount: 62 kg/ha."
        ],
        "soil_moisture": [
            "Maintain soil moisture at 70-80% during this stage.",
            "Keep soil consistently moist, around 75%.",
            "Water regularly to maintain 70% soil moisture.",
            "Maintain soil moisture at 68-78%.",
            "Keep soil moisture steady at 72%.",
            "Water moderately to maintain 70% moisture.",
            "Avoid water stress, keep soil moist at 74%.",
            "Ensure soil moisture is balanced at 73%.",
            "Maintain soil moisture at 69-79%.",
            "Keep soil moisture optimal between 70-80%.",
            "Maintain soil moisture at 71-81%.",
            "Keep soil moisture consistent at 76%.",
            "Water regularly to maintain 75% moisture.",
            "Avoid dry spells, keep soil moist at 70%.",
            "Ensure soil moisture is steady at 77%.",
            "Maintain soil moisture at 72-82%.",
            "Keep soil moisture balanced at 74%.",
            "Water moderately to maintain 73% moisture.",
            "Avoid waterlogging, keep soil moist at 75%.",
            "Maintain soil moisture at 70-80% with regular watering."
        ]
    },
    3: {
        "plant_info": "Stage 3: Flowering and fruiting stage - Development of flowers and fruits.",
        "fertilizer_recommendation": [
            "Use Phosphorus and Potassium rich fertilizer such as Diammonium Phosphate. Amount: 60 kg/ha.",
            "Apply potassium-rich fertilizer to support fruiting. Amount: 55 kg/ha.",
            "Supplement with bone meal for phosphorus. Amount: 50 kg/ha.",
            "Use potassium sulfate fertilizer. Amount: 58 kg/ha.",
            "Apply magnesium phosphate fertilizer. Amount: 53 kg/ha.",
            "Use rock phosphate fertilizer. Amount: 60 kg/ha.",
            "Apply bone meal fertilizer. Amount: 55 kg/ha.",
            "Use composted manure fertilizer. Amount: 52 kg/ha.",
            "Apply fish bone meal fertilizer. Amount: 50 kg/ha.",
            "Use kelp meal fertilizer. Amount: 54 kg/ha.",
            "Apply seaweed extract fertilizer. Amount: 56 kg/ha.",
            "Use humic acid fertilizer. Amount: 51 kg/ha.",
            "Apply fish emulsion fertilizer. Amount: 57 kg/ha.",
            "Use compost tea fertilizer. Amount: 53 kg/ha.",
            "Apply worm castings fertilizer. Amount: 55 kg/ha.",
            "Use alfalfa meal fertilizer. Amount: 52 kg/ha.",
            "Apply soybean meal fertilizer. Amount: 54 kg/ha.",
            "Use cottonseed meal fertilizer. Amount: 50 kg/ha.",
            "Apply blood meal fertilizer. Amount: 56 kg/ha.",
            "Use feather meal fertilizer. Amount: 53 kg/ha."
        ],
        "soil_moisture": [
            "Maintain soil moisture at 65-75% during this stage.",
            "Ensure soil moisture is steady around 70%.",
            "Avoid water stress by regular watering to 70% moisture.",
            "Maintain soil moisture at 63-73%.",
            "Keep soil moisture steady at 68%.",
            "Water regularly to maintain 70% moisture.",
            "Avoid dry spells, keep soil moist at 65%.",
            "Ensure soil moisture is balanced at 69%.",
            "Maintain soil moisture at 66-76%.",
            "Keep soil moisture optimal between 65-75%.",
            "Maintain soil moisture at 67-77%.",
            "Keep soil moisture consistent at 70%.",
            "Water regularly to maintain 72% moisture.",
            "Avoid water stress, keep soil moist at 68%.",
            "Ensure soil moisture is steady at 71%.",
            "Maintain soil moisture at 69-79%.",
            "Keep soil moisture balanced at 70%.",
            "Water moderately to maintain 73% moisture.",
            "Avoid waterlogging, keep soil moist at 72%.",
            "Maintain soil moisture at 65-75% with regular watering."
        ]
    }
}

main_folder = "C:\\Users\\hpcnd\\Desktop\\Harvestify\\"
#D:\\A DATA\\Project Module 2021-22\\A PYTHON New\\Plant Growth Prediction\\CompleteUpdatedPlantGrowth 
# Main function
def main(input_image):
    try:
        closest_stage = 0
        print(input_image)
        stages_resutl = {}
        images, filenames = load_images(main_folder+"static\\train")
        total_images = len(filenames)  # Dynamically set total images
        total_stages = 3

        # Compute base number of images per stage (some will have 4, some 5)
        images_per_stage = total_images // total_stages  # 4
        extra_images = total_images % total_stages  # 8 (first 8 stages get 5 images)
        image_names = [f"Img{i}.png" for i in range(1, total_images + 1)]
        stage_distribution = []
        start = 0
        for i in range(total_stages):
            num_images = images_per_stage + (1 if i < extra_images else 0)  # First 8 stages get 5 images, rest get 4
            stage_distribution.append(image_names[start:start + num_images])
            start += num_images

        # Reduce image vectors using PCA
        reduced_images, pca_model = reduce_dimensionality(images)
        
        # Load user-provided image
        user_img_path = main_folder+input_image
        user_img = cv2.imread(user_img_path, cv2.IMREAD_GRAYSCALE)
        user_img = cv2.resize(user_img, (100, 100)).flatten()
        user_img_vector = pca_model.transform(user_img.reshape(1, -1))  # Apply PCA transformation

        # Calculate similarity scores
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(user_img_vector.reshape(1, -1), reduced_images)[0]
        max_similarity = max(similarities)
        similarity_threshold = 0.7  # Threshold to determine if image is valid plant image

        if max_similarity < similarity_threshold:
            raise ValueError("Invalid Image: Not a valid plant image")

        closest_stage = find_closest_stage(input_image,user_img_vector, reduced_images, stage_distribution)
        if closest_stage+1==1:
            stages_resutl = {"Stage 1 Result:":random.choice(stage_distribution[0]),"Stage 2 will be:":random.choice(stage_distribution[1]),"Stage 3 Result:":random.choice(stage_distribution[2])}
        elif closest_stage+1==2:
            stages_resutl = {"Stage 2 Result:":random.choice(stage_distribution[1]),"Stage 3 will be:":random.choice(stage_distribution[2])}
        elif closest_stage+1==3:
            stages_resutl = {"Stage 3 Result:":random.choice(stage_distribution[2])}
        
        # Randomly select fertilizer and soil moisture info for each stage
        additional_info_dict = {}
        for stage, info in stage_info.items():
            additional_info_dict[stage] = {
                "plant_info": info["plant_info"],
                "fertilizer_recommendation": random.choice(info["fertilizer_recommendation"]),
                "soil_moisture": random.choice(info["soil_moisture"])
            }
        
    except Exception as e:
        print(e)
        additional_info_dict = {}
        # Propagate exception to caller
        raise e
    
    return closest_stage+1, stages_resutl, additional_info_dict

# print(main('static/input\Img20.png'))