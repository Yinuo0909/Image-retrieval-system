# Image-retrieval-system
The deployment of a medical image retrieval system involves many aspects, including database settings, image processing, environment configuration, etc. The following is user documentation for a medical image retrieval system that you can tailor to your specific technology stack and tools.
Demo video link : https://www.youtube.com/watch?v=08L2KUWaJNY

Step 1: System environment preparation
Install suitable Docker and PyCharm environments .

Step 2: Weaviate Database Setup
The system package contains a docker-compose.yml file that defines all the Weaviate modules required to run the demo. In this file, you will see this module, which in our case is the img2vec-neural module trained on the ResNet-50 model. To start a Weaviate instance, run the following command:


For details, please refer to :
https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/img2vec-neural
running Docker and the program, the login page will appear :
Username : admin Password : 12345

Step 3: Image processing
First create a folder to put your dataset images in .
(1) Run the defineSchema function in FTD_WEAVIATE.py to add the schema to the Weaviate instance .
(2) Use the prepared positive and negative sample data sets to upload to the Weaviate database. Since the image needs to be encoded into base64 using the blob dataType defined in the pattern, the png image needs to be encoded into base64 format data first.
(3) After you have defined the schema and converted the image to base64 value in the above steps, you can upload the data object to Weaviate.

As shown in the figure; after clicking the convert Base64 button, the png images of FTD positive samples and negative samples in the img folder will be converted into base64 encoded data and stored in the b64_img folder; after clicking the create Database button, the system will convert base6 Encoded data objects are uploaded to Weaviate. After running the define schema, convert Base64, and create Database buttons, the logs and admissions will print the relevant running logs, making it easier for users to use and manage the system.


Step 4: Image retrieval

Start by changing the number of samples in the code to the number you want to put in your gallery .
This module will use the nearImage operator in Weaviate so that it searches for the image that is closest to the image uploaded by the user. For this purpose, construct the search_similar_images function to obtain relevant results. The response to our search query will include the nearest objects in the FTD class. From the response, the function will output the thalamus diameter image with breed name and file path .
After adding the positive and negative sample images to be retrieved, the system enters the retrieval module. The system uses python language to call the weaviate client. When you click the Similarity Search button, a file selection box will pop up in the system interface, as shown in Figure 4.4. After selecting a thalamus diameter image to be retrieved through this selection box, the system will Schema Configuration is configured into Weaviate, and we will send it to the search_similar_images function. After the weaviate model algorithm, the system will automatically store the retrieved TOP2 results in the return object, and the decoded results will be displayed in the search page list.

Step 5: Algorithm Evaluation
This article conducted ten retrieval experiments under different maximum distances in the weaviate system by setting "distance" to 0:0.1:1, and calculated Precision and Recall respectively. Taking the precision rate and recall rate as the y-axis, distance is the x-axis, you can get the curve .
At the same time , the search results of each picture are recorded in the table to facilitate later calculation and analysis .
