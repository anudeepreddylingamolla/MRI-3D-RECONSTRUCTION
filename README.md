# MRI-3D-RECONSTRUCTION
The proposed approach involves using deep learning architectures, especially convolutional neural networks (CNNs), to achieve accurate and robust segmentation. Here's a high level overview for 3D MRI reconstruction.

1. Data Preprocessing and Dataset:
The dataset used for the project is BraTS2020 Dataset. BraTS has always been focusing on the evaluation of state-of-the-art methods for the segmentation of brain tumors in multimodal magnetic resonance imaging (MRI) scans. BraTS 2020 utilizes multi-institutional pre-operative MRI scans and primarily focuses on the segmentation of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, namely gliomas. Furthermore, to pinpoint the clinical relevance of this segmentation task, BraTS’20 also focuses on the prediction of patient overall survival, and the distinction between pseudoprogression and true tumor recurrence, via integrative analyses of radiomic features and machine learning algorithms. Finally, BraTS'20 intends to evaluate the algorithmic uncertainty in tumor segmentation .
	All BraTS multimodal scans are available as NIfTI files (.nii.gz) -> commonly used medical imaging format to store brain imaging data obtained using MRI and describe different MRI settings
1.	T1: T1-weighted, native image, sagittal or axial 2D acquisitions, with 1–6 mm slice thickness.
2.	T1c: T1-weighted, contrast-enhanced (Gadolinium) image, with 3D acquisition and 1 mm isotropic voxel size for most patients.
3.	T2: T2-weighted image, axial 2D acquisition, with 2–6 mm slice thickness.
4.	FLAIR: T2-weighted FLAIR image, axial, coronal, or sagittal 2D acquisitions, 2–6 mm slice thickness.
Data were acquired with different clinical protocols and various scanners from multiple (n=19) institutions.

	All the imaging datasets have been segmented manually, by one to four raters, following the same annotation protocol, and their annotations were approved by experienced neuro-radiologists. Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1), as described both in the BraTS 2012-2013 TMI paper and in the latest BraTS summarizing paper. The provided data are distributed after their pre-processing, i.e., co-registered to the same anatomical template, interpolated to the same resolution (1 mm^3) and skull-stripped.

2. Model Architecture:
U-Net, V-Net, or 3D versions of popular 2D architectures like ResNet or DenseNet can serve as the basis for our requirement. We have implemented the traditional U-Net architecture that is general used for the Biomedical Image Segmentation. The U-Net is convolutional network architecture for fast and precise segmentation of images.

 
Fig. 4.1 : U-Net Architecture

First sight, it has a “U” shape. The architecture is symmetric and consists of two major parts — the left part is called contracting path, which is constituted by the general convolutional process; the right part is expansive path, which is constituted by transposed 2d convolutional layers(you can think it as an upsampling technique for now). The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling. At each downsampling step we double the number of feature channels. Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”) that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU. The cropping is necessary due to the loss of border pixels in every convolution. At the final layer a 1x1 convolution is used to map each 64-component feature vector to the desired number of classes. In total the network has 23 convolutional layers.

3. Loss Function:
	The implemented loss function for the project is Dice loss. Dice Loss is a loss function commonly used in image segmentation tasks, particularly in medical image analysis and computer vision. It measures the similarity between the predicted segmentation mask and the ground truth mask by computing the overlap between the two masks in terms of the intersection and union of their pixels. Dice coefficient, which is essentially a measure of overlap between two samples. This measure ranges from 0 to 1 where a Dice coefficient of 1 denotes perfect and complete overlap. The Dice coefficient was originally developed for binary data, and can be calculated as:
 
In this formula, the intersection represents the number of common pixels between the predicted mask and the ground truth mask, while the union represents the total number of pixels in both masks. The Dice Loss aims to maximize the value when the predicted and ground truth masks are very similar, and minimize it when they differ significantly.Minimizing the Dice Loss helps in improving the accuracy and quality of the segmentation model's predictions, as it encourages better alignment between the predicted and actual segmentation boundaries.

 

4. Training:
We need to split the dataset into training, validation, and test sets. We train the model using the training data, optimizing the chosen loss function. Utilize the validation set to monitor training progress and prevent overfitting. Loading all data into memory is not a good idea since the data are too big to fit in. So we will create dataGenerators - load data on the fly. We have to keep in mind that in some cases, even the most state-of-the-art configuration won't have enough memory space to process the data the way we used to do it. That is the reason why we need to find other ways to do that task efficiently. By using data generator class, we are going to find an approach on how to generate our dataset on multiple cores in real time and feed it right away to our deep learning model.
A data generator is a technique commonly used in machine learning and deep learning to efficiently feed data to a model during training. Instead of loading and storing all the data in memory, which can be memory-intensive and slow for large datasets, a data generator generates batches of data on-the-fly and feeds them to the model as needed.
Here's a brief explanation of how a data generator works:
●	Data Loading: The data generator accesses the raw data from a source, such as files on disk or a database.
●	Batch Generation: It divides the data into smaller batches. A batch is a subset of the entire dataset that can fit into memory. Each batch contains a number of input examples (data points) and their corresponding target labels (if applicable).
●	Data Preprocessing: The generator applies any necessary data preprocessing steps to the batch. This might include tasks like resizing images, normalizing pixel values, data augmentation (randomly transforming input data to increase variety), and more.
●	Yielding Batches: During training, the data generator yields one batch at a time to the model. The model processes the batch, computes gradients, and updates its internal parameters through backpropagation.
●	Memory Efficiency: After a batch is processed, the generator can discard it from memory, freeing up space for the next batch. This is particularly useful when dealing with large datasets that wouldn't fit entirely in memory.
●	Epoch Completion: Once all batches have been processed and provided to the model, an epoch is completed. The generator can reshuffle the data or apply other strategies to ensure randomness and diversity in each epoch.

 
Fig. 4.2 : Data Distribution
	

5. Model Fitting:
After the splitting of data into test, train and validation we feed the model with the train and test the model using test and validation data. Here as we use the data generator class for feeding the data to the model, separate data generators for each train, test and validation data. We add the callback for the training process. Callbacks are a crucial component in the training process of machine learning models, particularly in deep learning frameworks like TensorFlow and Keras. They allow you to customize and control the behavior of the training loop by adding specific functions or actions at various points during training. Callbacks enable you to monitor, influence, or respond to the training process without manually intervening. 
CSV Logger is a callback function commonly used in machine learning frameworks like Keras to track and log training metrics during the training process. It saves the training history, including metrics like loss and accuracy, into a Comma-Separated Values (CSV) file. This file can be easily opened and analyzed using spreadsheet software or other data analysis tools. CSV Logger is a callback function commonly used in machine learning frameworks like Keras to track and log training metrics during the training process. It saves the training history, including metrics like loss and accuracy, into a Comma-Separated Values (CSV) file. This file can be easily opened and analyzed using spreadsheet software or other data analysis tools.

6. Evaluation:
We evaluate the trained model on the test dataset using metrics like Dice coefficient, Jaccard index, sensitivity, specificity, and Hausdorff distance. After the model is ready to predict the segmentation of the tumor, we visualize the segmentation results alongside the original MRI images for qualitative assessment. Various regions of the tumor like core, enhancing and whole tumor can be obtained from the 3D MRI scan. Since there can be anytype of MRI scan that can be used for the segmentation of tumor, the model detects the exact portion of the tumor and provides as output for scans like T1, T1c, T2, Flair.
Segmentation of gliomas in pre-operative MRI scans can be found on the principle as mentioned below:
Each pixel on image must be labeled:
●	Pixel is part of a tumor area (1 or 2 or 3) -> can be one of multiple classes / sub-regions
●	Anything else -> pixel is not on a tumor region (0)
The sub-regions of tumor considered for evaluation are: 1) the "enhancing tumor" (ET), 2) the "tumor core" (TC), and 3) the "whole tumor" (WT) The provided segmentation labels have values of 1 for NCR & NET, 2 for ED, 4 for ET, and 0 for everything else.
 
Fig. 4.3 : Different Parts of the Tumor 





7. 3D Reconstruction:
After segmentation of the tumor from the 3D MRI we need to produce the 3D view of the obtained glioma. This can be achieved by the method of converting the obtained image of the glioma into a 3D mesh. For this we need to apply the marching cubes algorithms for obtaining the required edges and vertices that are required for generation of the 3D mesh.
		Marching Cubes is an algorithm used in computer graphics and scientific visualization to create three-dimensional surfaces (meshes) from volumetric data, typically obtained from medical imaging or scientific simulations. The algorithm converts volumetric data, often represented as a 3D grid of scalar values or intensities, into a polygonal mesh that represents the outer surface of the object or region of interest. Here's a brief explanation of how Marching Cubes works:
●	Sampling Grid: The volumetric data is divided into a grid of small cubic cells. Each cell contains scalar values representing properties like density or intensity at its vertices.
●	Isovalue Thresholding: The algorithm defines an isovalue threshold. Vertices of the cells that cross this threshold are classified as inside or outside the object of interest.
●	Surface Extraction: For each cell, Marching Cubes determines the configuration of vertices that are inside and outside the object based on the isovalue threshold. This configuration is then used to generate polygons (usually triangles) that approximate the surface of the object within that cell.
●	Mesh Creation: The generated polygons from all the cells are combined to create a continuous mesh that approximates the surface of the object. Adjacent polygons are connected to form a coherent surface representation.
●	Smoothing and Refinement (Optional): Depending on the application, additional post-processing steps like smoothing or mesh simplification may be applied to improve the visual quality or reduce the complexity of the resulting mesh.
	After obtaining the vertices and faces from the marching cubes we provide them as input to the mesh library for obtaining the 3D mesh. In Python, a "mesh" refers to a data structure that represents a three-dimensional surface or object by defining its geometry using vertices, edges, and faces (typically triangles). It's commonly used in computer graphics, scientific visualization, and simulation. Some common parameters are:
Vertices (vertices): Vertices are the points in 3D space that define the shape of the mesh. Each vertex is represented as a coordinate (x, y, z). Vertices are stored in a numpy array or a similar data structure.
Faces (faces or triangles): Faces are the flat polygons that make up the surface of the mesh. In most cases, triangles are used because they are simple and versatile. Each face is defined by specifying the indices of the vertices that form the corners of the triangle.
Normals (normals, optional): Normals are vectors that point outward from each vertex and are used to determine the orientation of the surface at that point. Normals are important for lighting calculations and rendering. They can be manually computed or automatically generated.
Texture Coordinates (texcoords, optional): Texture coordinates are used to map 2D texture images onto the 3D surface of the mesh. They define how the texture is applied to the vertices.
