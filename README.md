# 💖 Face Mask Detection 
 ## 😷 Face Mask Recognition System Using Flask and Tensorflow
 This is project under under CSO 211(COMPUTER SYSTEM ORGANIZATION) aimed at using Computer Vision to detect face mask in real time.The detector created is pretty   accurate, and since we used the MobileNetV2 architecture, it’s also computationally efficient and can be used in embeded systems. The project runs on the server whose backend is built using Flask.
 This project can be integrated with embedded systems for application in airports, railway stations, offices, schools, and public places to ensure that public safety guidelines are followed.
 
 ## ⚙️ Project Structure
  #### The structure of the project was as follows -
  1. The project uses caffe based implementation for face and gender detection . The gender detection model is work by Gil Levi and Tal Hassner. Link to the [research paper](https://talhassner.github.io/home/publication/2015_CVPR)
  2. A model was trained on the dataset to sucessfully detect mask on the face using Keras 
  3. The project was then deployed using Flask to run on a webserver.
  4. To improve the fps we have used a custom based I/O technique using threading and added FPS count implementation to calculate fps.

 ## 💻Features
 1. Real Time Camera
 2. Responsive UI.
 3. Face Recognition in the frame
 4. Mask Detection on Faces in the frame
 5. If the person in frame is not wearing mask the app will also respond with a beep sound.
 6. Gender Detection on Faces in the frame
 7. Mask Detection Model 98% accurate
 8. On clicking the video button, mask detector model gets activated which can be seen through a window which will be opened up on clicking the button, displaying the     results of the model. I have used the deep learning pre-trained model to detect the faces from the image of different angles, and the model that used is Caffe         model.
    This functionality also predicts the gender of the person in the frame along with detecting face mask.The prediction that the person is wearing mask or not was 
    achieved through the application of convolutional neural networks. The model was trained in the keras environment.
    
 ## 🔑 Characteristcs:
 1. The detector uses a face detector to first get the bounding box for the face.
 2. After getting the boundaries of the face, we can pass it into our face mask detector which we are going to build using the TensorFlow/Keras environment.
 3. The face detector model was constructed using mobilenetv2 architecture as the base model(feature extractor) and adding a few layers prior to get the softmax           predictions.
 4. The loss function used was binary crossentroppy and model was trained using adam optimizer with standard hyperparameters.
 5. The mask detector was able to reach 98% accuracy with 0.93 f1 score on validation set.  


 ## ⚠️ Tech Stack Used
 [![My Skills](https://skills.thijs.gg/icons?i=python,js,html,css,tensorflow,flask)](https://skills.thijs.gg)
 1. Python 
 2. Tensorflow
 3. Flask
 4. Keras
 5. HTML
 6. CSS
 7. Javascript
 ## ⌛ Accuracy Plot
 <br> <img height="500" width="700" src="https://github.com/rakhi786/Engage-2022/blob/main/Accuracy_Plot/Capture.PNG"><br>

  
  ## 🚀&nbsp; Installation
1. Clone the repo
```
$ git clone https://github.com/rakhi786/Engage-2022.git
```

2. Change your directory to the cloned repo and create a Python virtual environment named 'venv' or anything you wish. Prefer this option if you want to directly test it using the terminal otherwise install all the dependencies directly on you machine 
3. Preferable Use Linux based OS or in Windows use Anaconda Prompt ([link to install Anaconda](https://www.anaconda.com/products/distribution) ) . To install virtual environment on Windows use the following command
```
$ pip install virtualenv
```
4. To Start Virtual Environment on Windows (use Anaconda Prompt )
```
$ virtualenv venv
$ .\venv\Scripts\activate

```
5. To start Virtual Environment on Linux based OS
```
$ sudo apt install python3-venv
$ python3 -m venv venv
$ source venv/bin/activate
```
6. Linux Users make sure LibGL library is already installed on your system as it is OpenCV dependency. To install it run the following command 
   in your terminal before starting the virtual environment
 ```
 $ sudo apt-get update
 $ sudo apt install -y libgl1-mesa-glx
 ```
7. Now your virtual environment will start. Run the following command to install all the dependencies. Linux Users use pip3 command instead of pip.
```
$ pip install -r requirements.txt
```
7. Run the following command to start the server 
```
$ flask run
```
8. Now Head onto the following server [http://127.0.0.1:5000/](http://127.0.0.1:5000/) or the server mentioned in the command line to access the webpage .

### ⌛ Working
   On opening the server, we can click on the video button to get the expected results. On clicking the video button,mask detector model gets activated which can be      seen through a window which will be opened up on clicking the button, displaying the results of the model. WE CAN DESTROY THE OPENED WINDOW BY PRESSING 'Q'.Then we    will again redirected to our home page.
 
### Optional Features
#### Frame Rate Per Second 
   1. To analyze frames rate per second run the following command. Linux users use python3 instead of python in the terminal
   ```
   $ python Fps_Analysis.py  
   ```
#### Training 
   1. If you want to train the model using your own dataset run the following command. Linux users use python3 instead of python in the terminal
   ```
   $ python train.py --datatset <name of the dataset folder>  
   ```
   
 
> **Feel free to contribute ✨**.   
