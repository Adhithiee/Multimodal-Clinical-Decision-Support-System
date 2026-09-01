import os
import pandas as pd
from tqdm import tqdm
from preprocess_images import preprocess_single_image
from preprocess_text import preprocess_reports_df

def run_pipeline():
    # Define filepaths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_csv = os.path.join(base_dir, "indiana_reports.csv")
    projections_csv = os.path.join(base_dir, "indiana_projections.csv")
    raw_img_dir = os.path.join(base_dir, "images", "images_normalized")
    processed_img_dir = os.path.join(base_dir, "images", "images_processed")
    
    print("Step 1: Preprocessing and cleaning text reports...")
    reports_df = preprocess_reports_df(reports_csv)
    
    print("Step 2: Loading and linking projections (image mapping)...")
    projections_df = pd.read_csv(projections_csv)
    
    # Group projection images by patient UID
    linked_images = {}
    for _, row in projections_df.iterrows():
        uid = int(row['uid'])
        filename = row['filename']
        projection = row['projection'].strip().lower() # Frontal or Lateral
        
        if uid not in linked_images:
            linked_images[uid] = {'frontal': [], 'lateral': []}
            
        if 'frontal' in projection:
            linked_images[uid]['frontal'].append(filename)
        elif 'lateral' in projection:
            linked_images[uid]['lateral'].append(filename)
            
    # Step 3: Run pipeline and copy/preprocess images
    print("Step 3: Processing images (Grayscale + CLAHE + Resizing)...")
    os.makedirs(processed_img_dir, exist_ok=True)
    
    processed_records = []
    
    for _, row in tqdm(reports_df.iterrows(), total=len(reports_df)):
        uid = int(row['uid'])
        
        # Check if patient has associated images
        if uid not in linked_images:
            continue
            
        patient_images = linked_images[uid]
        frontal_list = patient_images['frontal']
        lateral_list = patient_images['lateral']
        
        # We need at least one frontal or lateral image to include the case
        if len(frontal_list) == 0 and len(lateral_list) == 0:
            continue
            
        # Helper lists to store the local paths of preprocessed files
        processed_frontal = []
        processed_lateral = []
        
        # Preprocess Frontal Images
        for fname in frontal_list:
            raw_path = os.path.join(raw_img_dir, fname)
            proc_path = os.path.join(processed_img_dir, fname)
            
            if os.path.exists(raw_path):
                try:
                    preprocess_single_image(raw_path, proc_path)
                    # Store relative path for portability
                    processed_frontal.append(os.path.join("images", "processed", fname))
                except Exception as e:
                    print(f"Error processing image {fname}: {e}")
                    
        # Preprocess Lateral Images
        for fname in lateral_list:
            raw_path = os.path.join(raw_img_dir, fname)
            proc_path = os.path.join(processed_img_dir, fname)
            
            if os.path.exists(raw_path):
                try:
                    preprocess_single_image(raw_path, proc_path)
                    processed_lateral.append(os.path.join("images", "processed", fname))
                except Exception as e:
                    print(f"Error processing image {fname}: {e}")
                    
        # Only record if we successfully processed at least one image
        if len(processed_frontal) > 0 or len(processed_lateral) > 0:
            processed_records.append({
                'uid': uid,
                'findings': row['findings'],
                'impression': row['impression'],
                'indication': row['indication'],
                'mesh': row['mesh'],
                'problems': row['problems'],
                'frontal_images': ";".join(processed_frontal),
                'lateral_images': ";".join(processed_lateral)
            })
            
    # Save processed index mapping
    output_csv = os.path.join(base_dir, "processed_dataset.csv")
    output_df = pd.DataFrame(processed_records)
    output_df.to_csv(output_csv, index=False)
    
    print("\n--- Pipeline Completed Successfully ---")
    print(f"Total original patient reports: {len(reports_df)}")
    print(f"Total fully preprocessed patient records: {len(output_df)}")
    print(f"Output matched mapping saved to: {output_csv}")

if __name__ == "__main__":
    run_pipeline()
