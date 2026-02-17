import os
import csv
from PIL import Image
import numpy as np
from argparse import ArgumentParser

def analyze_anomaly_maps(directory='.', output_csv='anomaly_pixel_analysis.csv'):
    """
    Analyzes black and white anomaly maps and calculates white pixel counts and ratios.

    Args:
        directory: Directory containing the images
        output_csv: Output CSV file name
    """
    results = []

    # Supported image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    # Get all image files in directory
    image_files = [f for f in os.listdir(directory) 
                   if f.lower().endswith(image_extensions)]

    print(f"Found {len(image_files)} image files")

    for filename in sorted(image_files):
        filepath = os.path.join(directory, filename)

        try:
            # Open image and convert to grayscale
            img = Image.open(filepath).convert('L')
            img_array = np.array(img)

            # Calculate total pixels
            total_pixels = img_array.size

            # Count white pixels (assuming white is > 127 threshold)
            # Adjust threshold if your images use different convention
            white_pixels = np.sum(img_array > 127)

            # Calculate ratio
            white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0

            results.append({
                'filename': filename,
                'total_pixels': total_pixels,
                'white_pixels': white_pixels,
                'white_ratio': white_ratio,
                'white_percentage': white_ratio * 100,
                'image_width': img.width,
                'image_height': img.height
            })

            print(f"Processed: {filename} - {white_pixels}/{total_pixels} ({white_ratio:.4f})")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Write results to CSV
    if results:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'image_width', 'image_height', 'total_pixels', 
                         'white_pixels', 'white_ratio', 'white_percentage']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to {output_csv}")
        print(f"Total images processed: {len(results)}")
    else:
        print("No images were processed")

    return results


if __name__ == '__main__':
    # Run the analysis on current directory
   
    parser = ArgumentParser()
    parser.add_argument("--dirname", type=str, required=True)
    args = parser.parse_args()

    # Or specify a different directory:
    results = analyze_anomaly_maps(directory=args.dirname, 
                                   output_csv='can_anomalies.csv')
