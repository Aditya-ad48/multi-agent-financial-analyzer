from unstructured.partition.pdf import partition_pdf
import pandas as pd
import json
import os
import io
import pymupdf  # For extracting image files
from PIL import Image

def process_pdf_and_save_assets(
    pdf_path, 
    output_json="processed_data_final.json",
    csv_output_folder="tables_csv",
    image_output_folder="images"
):
    """
    Processes a PDF using unstructured, saving tables as CSVs and images as PNGs
    in separate folders, referenced by a main JSON file.
    """
    # Create output folders if they don't exist
    os.makedirs(csv_output_folder, exist_ok=True)
    os.makedirs(image_output_folder, exist_ok=True)

    print("Starting PDF processing with unstructured (hi_res strategy)... This may take a while.")
    # This single function call uses AI to partition the document
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True
    )

    doc = pymupdf.open(pdf_path)
    all_chunks = []
    
    for i, element in enumerate(elements):
        metadata = {
            "source": os.path.basename(pdf_path),
            "page": element.metadata.page_number,
            "type": element.category,
            "element_id": f"element_{i+1}"
        }
        content = ""
        
        # --- Handle Tables ---
        if element.category == "Table":
            try:
                df = pd.read_html(io.StringIO(element.metadata.text_as_html))[0]
                content = df.to_markdown(index=False)
                
                # Save the DataFrame to a CSV file
                csv_filename = f"table_page_{metadata['page']}_elem_{i+1}.csv"
                csv_path = os.path.join(csv_output_folder, csv_filename)
                df.to_csv(csv_path, index=False)
                metadata["csv_path"] = csv_path
                
            except Exception as e:
                content = f"[Could not parse table HTML: {e}]"
        
        # --- Handle Images ---
        elif element.category == "Image":
            try:
                page = doc[element.metadata.page_number - 1]
                coords = element.metadata.coordinates.points
                x1, y1 = coords[0]; x2, y2 = coords[2]
                
                # Use PyMuPDF to clip the image from the page
                image_bbox = pymupdf.Rect(x1, y1, x2, y2)
                pix = page.get_pixmap(clip=image_bbox, dpi=300)
                
                # Save the image as a PNG file
                image_filename = f"image_page_{metadata['page']}_elem_{i+1}.png"
                image_path = os.path.join(image_output_folder, image_filename)
                pix.save(image_path)
                
                content = f"[Image content saved to file: {image_path}]"
                metadata["image_path"] = image_path

            except Exception as e:
                content = f"[Error processing and saving image: {e}]"
        
        # --- Handle Text and Other Elements ---
        else:
            content = element.text

        all_chunks.append({
            "page_content": content,
            "metadata": metadata
        })
        
    doc.close()
    
    # Save the main JSON manifest file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)
        
    return all_chunks

if __name__ == "__main__":
    pdf_path = "/Users/aditya/Desktop/ML/INTERIIT-NLP/c87043b9-5d89-4717-9f49-c4f9663d0061.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' was not found.")
    else:
        structured_data = process_pdf_and_save_assets(pdf_path)
        
        if structured_data:
            text_count = sum(1 for item in structured_data if item['metadata']['type'] not in ['Table', 'Image'])
            table_count = sum(1 for item in structured_data if item['metadata']['type'] == 'Table')
            image_count = sum(1 for item in structured_data if item['metadata']['type'] == 'Image')
            
            print("\n--- Processing Complete ---")
            print(f"Total Elements Extracted: {len(structured_data)}")
            print(f"Text-based Elements: {text_count}")
            print(f"Table Elements: {table_count}")
            print(f"Image Elements: {image_count}")
            print(f"\n✅ All tables saved to the 'tables_csv' folder.")
            print(f"✅ All images saved to the 'images' folder.")
            print(f"✅ Main data manifest saved to 'processed_data_final.json'")