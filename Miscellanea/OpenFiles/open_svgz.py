import gzip
import cairosvg
from PIL import Image
import io

svgz_file = "CEG_graph.svgz"
output_file = "CEG_graph.jpg"

print("Reading and decompressing the file...")
# 1. Read and decompress the .svgz file into memory
with gzip.open(svgz_file, 'rb') as f:
    svg_data = f.read()

print("Converting to image (this might take a while or saturate the RAM)...")
# 2. Convert the SVG to PNG format in memory using Cairo
png_data = cairosvg.svg2png(bytestring=svg_data, dpi=300)

# 3. Open it with PIL to save it as JPG (since CairoSVG exports PNG)
image = Image.open(io.BytesIO(png_data))
# Convert to RGB because JPG does not support transparency
image_rgb = image.convert("RGB")
image_rgb.save(output_file, "JPEG", quality=95)

print(f"Done! Saved as {output_file}")