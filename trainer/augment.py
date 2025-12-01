
import os
import cv2 # type: ignore
import numpy as np # type: ignore
import albumentations as A # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

from configs.config import EXTRACT_PATH, AUGMENTED_PATH, COSINE_THRESHOLD, N_AUGMENTS
from recognizer.extractor import get_arcface_embedding

augment = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.MotionBlur(blur_limit=5, p=0.4),
    A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-20, 20), p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.4),
    A.HueSaturationValue(p=0.4),
    A.RandomGamma(p=0.4),
    A.ImageCompression(quality_range=(85, 100), p=0.4),
    A.CLAHE(p=0.3),
    A.Resize(160, 160),
])

def save_embedding(embedding, folder, filename):
    """Saves embedding as .npy file"""
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    npy_path = os.path.join(folder, f"{filename}.npy")
    np.save(npy_path, embedding)

def generate_valid_augments(img_path, save_dir, n_augments=N_AUGMENTS):
    """
    Generates augments, checks cosine similarity using external extractor, 
    and saves .npy if valid.
    Returns list of valid embeddings (including original).
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return []

    orig_emb = get_arcface_embedding(img_bgr)
    
    if orig_emb is None or len(orig_emb) == 0:
        print(f"Skipping {os.path.basename(img_path)}: No face detected in original.")
        return []
    
    orig_emb = np.array(orig_emb).flatten()

    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    save_embedding(orig_emb, save_dir, base_name)
    
    valid_embeddings = [orig_emb] 
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    count = 0
    attempts = 0
    max_attempts = n_augments * 3  # Prevent infinite loops
    
    while count < n_augments and attempts < max_attempts:
        attempts += 1
        
        augmented_rgb = augment(image=img_rgb)['image']
        
        augmented_bgr = cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2BGR)
        
        aug_emb = get_arcface_embedding(augmented_bgr)
        
        if aug_emb is None or len(aug_emb) == 0:
            continue
            
        aug_emb = np.array(aug_emb).flatten()
        
        sim = cosine_similarity(orig_emb.reshape(1, -1), aug_emb.reshape(1, -1))[0][0]
        
        if sim > COSINE_THRESHOLD:
            count += 1
            filename = f"{base_name}_aug{count}"
            
            # cv2.imwrite(os.path.join(save_dir, f"{filename}.jpg"), augmented_bgr)
            save_embedding(aug_emb, save_dir, filename)
            valid_embeddings.append(aug_emb)
            
    return valid_embeddings

if __name__ == "__main__":
    if not os.path.exists(EXTRACT_PATH):
        print(f"Error: Path {EXTRACT_PATH} does not exist.")
        exit()

    people = sorted([
        p for p in os.listdir(EXTRACT_PATH)
        if os.path.isdir(os.path.join(EXTRACT_PATH, p))
    ])

    print(f"Total people found: {len(people)}")

    for person in people:
        person_path = os.path.join(EXTRACT_PATH, person)
        
        img_names = sorted([
            f for f in os.listdir(person_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        # Save to AUGMENTED_PATH/{person_name}
        save_dir = os.path.join(AUGMENTED_PATH, person)
        
        all_person_embeddings = []

        for img_name in img_names:
            img_path = os.path.join(person_path, img_name)
            
            embeddings = generate_valid_augments(
                img_path, 
                save_dir, 
                n_augments=N_AUGMENTS
            )
            
            if embeddings:
                all_person_embeddings.extend(embeddings)

        if len(all_person_embeddings) > 0:
            emb_matrix = np.array(all_person_embeddings)
            
            # mean_emb = np.mean(emb_matrix, axis=0)
            
            # Normalize the mean embedding
            # mean_emb = mean_emb / np.linalg.norm(mean_emb) + 1e-12
            mean_emb = np.median(emb_matrix, axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)

            
            filename = f"{person}_average_embedding"
            save_embedding(mean_emb, save_dir, filename)
            # print(f" -> Saved average embedding for {person}")

    print("Processing Complete.")