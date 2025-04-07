import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from scipy import ndimage
from skimage.morphology import binary_closing, ball
import math
import trimesh  # Import the trimesh library
from skimage import measure

# CONFIGURATION
your_directory = "C:/Users/taral/Documents/vscode/data_and_image_files/mri"
initial_slice_name = "1-033.dcm"
slice_file_prefix = "1-"
slice_file_extension = ".dcm"
slices_sub_directory = ""

# LOAD REFERENCE SLICE
initial_slice_path = os.path.join(your_directory, initial_slice_name)
RefDs = pydicom.dcmread(initial_slice_path)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns))
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
ArrayDicom[:, :] = RefDs.pixel_array

# SLICE SELECTION VARIABLES
InFile = 100
FinalFile = 192
StartSlice = 118
EndSlice = 148

assert InFile <= StartSlice <= EndSlice <= FinalFile, "Slice range is outside bounds."

# LOAD DICOM SLICES
SliceCount = EndSlice - StartSlice + 1
ArrayDicom3D = np.zeros((SliceCount, RefDs.Rows, RefDs.Columns), dtype=RefDs.pixel_array.dtype)

for i, ix in enumerate(range(StartSlice, EndSlice + 1)):
    slice_filename = f'{slice_file_prefix}{ix:03d}{slice_file_extension}'
    slice_path = os.path.join(your_directory, slices_sub_directory, slice_filename)
    RefDs = pydicom.dcmread(slice_path)
    ArrayDicom3D[i, :, :] = RefDs.pixel_array

# SMOOTHING AND THRESHOLDING
smoothed3D = ndimage.gaussian_filter(ArrayDicom3D.astype(np.float32), sigma=1)
Threshold = 600
mask3D = (smoothed3D > Threshold).astype(np.uint8)

for ix in range(mask3D.shape[0]):
    mask3D[ix] = ndimage.binary_fill_holes(mask3D[ix]).astype(np.uint8)

mask3D = binary_closing(mask3D, ball(2))

# ISOLATE CENTRAL TUMOUR
labeled_array, num_features = ndimage.label(mask3D)
sizes = ndimage.sum(mask3D, labeled_array, range(num_features + 1))

center = np.array(mask3D.shape) // 2
max_size = 0
best_label = 0

for label in range(1, num_features + 1):
    coords = np.argwhere(labeled_array == label)
    if len(coords) == 0:
        continue
    centroid = coords.mean(axis=0)
    distance = np.linalg.norm(centroid - center)
    size = sizes[label]
    if size > max_size and distance < 80:
        max_size = size
        best_label = label

tumour_mask = (labeled_array == best_label).astype(np.uint8)

# FILL INTERNAL 3D GAPS
def fill_3d_holes(binary_mask):
    inverse = 1 - binary_mask
    filled = ndimage.binary_fill_holes(inverse)
    internal_holes = filled ^ inverse
    return binary_mask | internal_holes

tumour_mask = fill_3d_holes(tumour_mask).astype(np.uint8)

# FILLED VOLUME (for display or scalar mesh)
SegmentedArrayDicom3D = tumour_mask * ArrayDicom3D

# VOXELIZED STL EXPORT WITH TRimesh AND LAPLACIAN SMOOTHING
def voxel_to_smoothed_mesh(mask3D_input, filename="tumour_smoothed_trimesh.stl", voxel_size=1.0, smoothing_iterations=10, smoothing_factor=0.1):
    """Create an STL mesh from voxels using trimesh and apply Laplacian smoothing."""

    # Use marching cubes to create a surface mesh from the volume
    vertices, faces, _, _ = measure.marching_cubes(mask3D_input, level=0.5, spacing=(voxel_size, voxel_size, voxel_size))

    if len(vertices) == 0 or len(faces) == 0:
        print("Warning: No surface found after marching cubes. Cannot perform smoothing.")
        return

    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Apply Laplacian smoothing
    smoothed_mesh = mesh.copy()
    for _ in range(smoothing_iterations):
        smoothed_mesh = trimesh.smoothing.filter_laplacian(smoothed_mesh, iterations=1, lamb=smoothing_factor) # Corrected keyword: lamb

    # Save the smoothed mesh
    smoothed_mesh.export(filename)
    print(f"Smoothed STL mesh (using trimesh) saved: {filename}")

# Generate and save the smoothed mesh using trimesh
voxel_to_smoothed_mesh(tumour_mask, filename="tumour_smoothed_trimesh.stl")

# EXTRAS
# Cross-slice inspection: View middle slice (optional)
mid_slice = tumour_mask.shape[0] // 2
plt.imshow(SegmentedArrayDicom3D[mid_slice], cmap="gray")
plt.title("Mid Slice - Tumour Only (Filled)")
plt.show()

# Display each image with the 3d mask
num_slices = mask3D.shape[0]
# Calculate a reasonable number of rows and columns for the subplots
nrows = int(math.sqrt(num_slices))
ncols = math.ceil(num_slices / nrows)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 16))
axes = axes.flatten()  # Flatten to make indexing easier
# Loop through and plot each image
for i in range(num_slices):
    ax = axes[i]
    ax.pcolormesh(np.flipud(mask3D[i]), cmap='gray')
    ax.set_title(f"ID: {i+StartSlice}")  # Title with original slice ID
    ax.axis('equal')  # Make the image square
    ax.axis('off')  # Turn off axis labels
    ax.set_xticks([])
    ax.set_yticks([])
# Remove any unused subplots if num_slices is less than the total number of axes
if num_slices < len(axes):
    for j in range(num_slices, len(axes)):
        fig.delaxes(axes[j])
plt.subplots_adjust(wspace=0.1, hspace=0.4)
plt.show()