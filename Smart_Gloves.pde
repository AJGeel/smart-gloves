//*********************************************
// Time-Series Signal Processing and Classification
// e10_LoadSVM_Gesture_Arduino_ThreeSensors
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************
// The papaya library is included in the /code folder.
// Papaya: A Statistics Library for Processing.Org
// http://adilapapaya.com/papayastatistics/
// Before use, please make sure your Arduino has e sensor connected
// to the analog input, and SerialString_ThreeSensors.ino was uploaded. 
//[0-9] Change Label to 0-9
//[ENTER] Train the SVM
//[TAB] Increase Label Number
//[/] Clear the Data
// [SPACE] Pause Data Stream
// [A] Increase the Activation Threshold by 10
// [Z] Decrease the Activation Threshold by 10

import papaya.*; //statistic library for processing
import processing.serial.*;
Serial port; 

int sensorNum = 3; //number of sensors in use
int dataNum = 500; //number of data to show
int[] rawData = new int[sensorNum]; //raw data from serial port
float[] postProcessedDataArray = new float[sensorNum]; //data after postProcessing
float[][] sensorHist = new float[sensorNum][dataNum]; //history data to show
boolean b_pause = false; //flag to pause data collection

float[][] diffArray = new float[sensorNum][dataNum]; //diff calculation: substract

float[] modeArray = new float[dataNum]; //To show activated or not
int activationThld = 7; //The diff threshold of activiation

int windowSize = 180; //The size of data window
float[][] windowArray = new float[sensorNum][windowSize]; //data window collection
boolean b_sampling = false; //flag to keep data collection non-preemptive
int sampleCnt = 0; //counter of samples

//Statistical Features
float[] windowM = new float[sensorNum]; //mean
float[] windowSD = new float[sensorNum]; //standard deviation
float[] windowMax = new float[sensorNum]; //max
float[] windowMin = new float[sensorNum]; //min
boolean b_featureReady = false;

//SVM parameters
double C = 64; //Cost: The regularization parameter of SVM
int d = 12;     //Number of features to feed into the SVM
int lastPredY = -1;

//Custom parameters
int repCount = 0;
int MAX_REPS = 5;
boolean switchToNextExercise = false;
int currentExercise = 0;
boolean started = false;
int currentY = 0;

// ------------ //
// Images / GUI //
// ------------ //
/* Colours and images */
PImage img0;
PImage imgAGood;
PImage imgABad;
PImage imgBGood;
PImage imgBBad;
PImage imgCGood;
PImage imgCBad;

color lightGray = color(178,174,201);
color deepGray = color(40);
color deepRed = color(246,110,98);
color black = color(0);

int imgWidth = 550;
int imgHeight = 216;
int smallText = 16;
int bigText = 28;
int padding = 60;

/* Global variables */
String repCountStr = ("0" + repCount);
String stylizedRepCount = repCountStr.substring(repCountStr.length()-2); // Adds a 0 in front of numbers 0 to 9.


String[] exercises = {"Biceps Curl", "Lateral Raise", "Arnold Press"}; // Array with the names of all exercises
int exCount = 0; // Number that retrieves the correct exercise from the array 

void setup() {
  size(800, 500);
  background(255);
  img0 = loadImage("img0.jpg");
  imgAGood = loadImage("img-a-good.jpg");
  imgABad = loadImage("img-a-bad.jpg");
  imgBGood = loadImage("img-b-good.jpg");
  imgBBad = loadImage("img-b-bad.jpg");
  imgCGood = loadImage("img-c-good.jpg");
  imgCBad = loadImage("img-c-bad.jpg");

  //Initiate the serial port
  for (int i = 0; i < Serial.list().length; i++) println("[", i, "]:", Serial.list()[i]);
  String portName = Serial.list()[Serial.list().length-1];//check the printed list
  port = new Serial(this, portName, 115200);
  port.bufferUntil('\n'); // arduino ends each data packet with a carriage return 
  port.clear();           // flush the Serial buffer

  for (int i = 0; i < modeArray.length; i++) { //Initialize all modes as null
    modeArray[i] = -1;
  }
  showScreen(0); // default state
  
  model = loadSVM_Model(sketchPath()+"/data/test.model"); //load an SVM here
  svmTrained = true; //set true because it is loaded.
  firstTrained = false; //set false because it is not the first training.
}

void draw() {
  //background(255);
  
  if (!svmTrained && firstTrained) {
    //train a linear support vector classifier (SVC) 
    trainLinearSVC(d, C);
  }
  
  repCountStr = ("0" + repCount);
  stylizedRepCount = repCountStr.substring(repCountStr.length()-2); // Adds a 0 in front of numbers 0 to 9.
  showScreen(currentY);
 
  
  //Draw the sensor data
  //lineGraph(float[] data, float lowerbound, float upperbound, float x, float y, float width, float height, int _index)
  //for (int i = 0; i < sensorNum; i++) {
  //  lineGraph(sensorHist[i], 0, height, 0, 0, width, height*0.3, i); //history of signal
  //  lineGraph(diffArray[i], -height, height, 0, height*0.3, width, height*0.3, i); //history of diff
  //  lineGraph(windowArray[i], 0, height, 0, height*0.6, width, height*0.3, i); //history of window
  //}

  ////Draw the modeArray
  ////barGraph(float[] data, float lowerbound, float upperbound, float x, float y, float width, float height)
  //barGraph(modeArray, -1, 1, 0, height, width, height*0.1);

  //pushStyle();
  //fill(0);
  //textSize(18);
  //text("Activation Threshold: "+activationThld, 20, 40);
  //for (int i = 0; i < sensorNum; i++) {
  //  text("["+i+"] M: "+windowM[i]+", SD: "+windowSD[i]+", Max: "+windowMax[i]+", Min: "+windowMin[i], 20, 70+i*30);
  //}
  //if(svmTrained)
  //text("Last Prediction:"+lastPredY, 20, width-60);
  //popStyle();
}

void serialEvent(Serial port) {   
  String inData = port.readStringUntil('\n');  // read the serial string until seeing a carriage return
  int dataIndex = -1;
  if (!b_pause) {
    //assign data index based on the header
    if (inData.charAt(0) == 'A') {  
      dataIndex = 0;
    }
    if (inData.charAt(0) == 'B') {  
      dataIndex = 1;
    }
    if (inData.charAt(0) == 'C') {  
      dataIndex = 2;
    }
    //data processing
    if (dataIndex>=0) {
      rawData[dataIndex] = int(trim(inData.substring(1))); //store the value
      postProcessedDataArray[dataIndex] = map(constrain(rawData[dataIndex], 0, 1023), 0, 1023, 0, height); //scale the data (for visualization)
      appendArray(sensorHist[dataIndex], rawData[dataIndex]); //store the data to history (for visualization)
      float diff = abs(sensorHist[dataIndex][0] - sensorHist[dataIndex][1]); //absolute diff is used
      appendArray(diffArray[dataIndex], diff); //store the abs diff to history (for visualization)
      
   

      if (diff>activationThld) { //activate when the absolute diff is beyond the activationThld
        if (b_sampling == false) { //if not sampling
          println("Starting!");
          b_sampling = true; //do sampling
          sampleCnt = 0; //reset the counter
          for (int i = 0; i < sensorNum; i++) {
            for (int j = 0; j < windowSize; j++) {
              windowArray[i][j] = 0; //reset the window
            }
          }
        }
      }

      if (b_sampling == true) {
        appendArray(windowArray[dataIndex], rawData[dataIndex]); //store the windowed data to history (for visualization)
        ++sampleCnt;
        if (sampleCnt == (windowSize*sensorNum)) {
          for (int i = 0; i < sensorNum; i++) {
            windowM[i] = Descriptive.mean(windowArray[i]); //mean
            windowSD[i] = Descriptive.std(windowArray[i], true); //standard deviation
            windowMax[i] = Descriptive.max(windowArray[i]); //max
            windowMin[i] = Descriptive.min(windowArray[i]); //min
          }
          b_sampling = false; //stop sampling if the counter is equal to the window size
          b_featureReady = true;
        }
      }

      //use the data for classification
      if (dataIndex == 2) { // if the header is 'C', we collect the data as a set 
        if (b_featureReady == true) {
          float[] mean = new float[sensorNum];
          float[] sd = new float[sensorNum];
          float[] min = new float[sensorNum];
          float[] max = new float[sensorNum];
          for (int i = 0; i < postProcessedDataArray.length; i ++) {
            mean[i] = windowM[i];
            sd[i] = windowSD[i];
            min[i] = windowMin[i];
            max[i] = windowMax[i]; 
          }
          if (!svmTrained) { //if the SVM model is not trained
            int Y = type; //Form a label Y;
            //double[] dataToTrain = {X[0], X[1], X[2], Y}; //Form a dataToTrain with label Y
            //trainData.add(new Data(dataToTrain)); //Add the dataToTrain to the trainingData collection.
            appendArray(modeArray, Y); //append the label to  for visualization
            println("train:",Y);
            ++tCnt;
          } else { //if the SVM model is trained
            float xMean = mean[0];
            float yMean = mean[1];
            float zMean = mean[2];
            float xSD   = sd[0];
            float ySD   = sd[1];
            float zSD   = sd[2];
            double[] dataToTest = {xMean, yMean, zMean, xSD, ySD, zSD, min[0], min[1], min[2], max[0], max[1], max[2]}; //Form a dataToTest without label Y
            int predictedY = (int) svmPredict(dataToTest); //SVMPredict the label of the dataToTest
            lastPredY = predictedY; //update the lastPredY;
            appendArray(modeArray, predictedY); //append the prediction results to modeArray for visualization
            int predictedType = predictedY/10; //implicit integer conversion to get first number
            //currentExercise = predictedType;
            if(!started) {  //If first time running allow mistakes to count
              switch(predictedType) {
                case 1:
                  predictedY = 10;
                  currentExercise = 1;
                  break;
                case 2:
                  predictedY = 20;
                  currentExercise = 2;
                  break;
                case 3:
                  predictedY = 30;
                  currentExercise = 3;
                  break;
              }
              started = true;
            }
            
            if(currentExercise == 3){
            }
            if(currentExercise == predictedType){
              currentY = predictedY;
              if(predictedY%10 == 0 && started) { // If exercise is done correctly
                repCount++;
              }
            } else {
              currentY = (currentExercise*10) + 1;
            }
            
            if(repCount == MAX_REPS) {
              delay(1000); // Small delay to show 10 exercise being done
              // Fill in if set count has been completed
              repCount = 0;
              currentY = 30;
              switchToNextExercise();                
            }
            
            //showScreen(predictedY);
            draw();
            println("Done measuring:",predictedY,millis());
          }
          b_featureReady = false; //reset the flag for the next update
        } else {
          if (!svmTrained) { //if the SVM model is not trained
            appendArray(modeArray, -1); //the class is null without mouse pressed.
          }else{
            appendArray(modeArray, lastPredY); //append the last prediction results to modeArray for visualization
          }
        }
      }
      return;
    }
  }
}

void switchToNextExercise() {
  // Get next exercise it needs to switch to
  println("switching", currentExercise);
  switch(currentExercise) {
    case 1:
      currentExercise = 2;
      currentY = 20;
      draw();
      println("next",currentY);
      break;
    case 2:
      currentExercise = 3;
      currentY = 30;
      draw();
      println("next",currentY);
      break;
    case 3:
      currentExercise = 1;
      currentY = 10;
      draw();
      println("next",currentY);
      break;
    default:
      println("Error: couldnt match");
  }
  // Choose image based on exercise
  // Switch old image with new image
  // Set variables (setcounter/repcounter/which one is next) to 0
}

void showScreen(int state) {
  if (state == 0){          //        -----------------------------------------------------------   STARTING SCREEN   --------------------------------------------------------------------------------------
  textAlign(CENTER);
  background(255);
  
  /* Top text */
  textSize(bigText*1.5);
  fill(deepGray);
  text("Let's Exercise!", width/2, padding*1.4);
  fill(lightGray);
  textSize(smallText*1.5);
  text("Why not start with one of these?", width/2, padding*2);
  
  
  fill(black);
  text(exercises[0], .2*width, height-padding*.8);
  text(exercises[1], .5*width, height-padding*.8);
  text(exercises[2], .8*width, height-padding*.8);
 
  
  /* Central image */
  imageMode(CENTER);
  image(img0, width/2, height*.58, 700, 250);

  }
  
  
  if (state == 10){          //        -----------------------------------------------------------   EXERCISE A: GOOD FORM   --------------------------------------------------------------------------------------
  textAlign(CENTER);
  background(255);
  
  /* Top text */
  textSize(smallText);
  fill(lightGray);
  text("CURRENT EXERCISE", width/2, padding/2);
  textSize(bigText+bigText/5);
  fill(black);
  text(exercises[exCount], width/2, padding+padding/8);
  
  /* Bottom left text*/
  textAlign(LEFT);
  textSize(smallText);
  fill(lightGray);
  text("REP COUNT", 50, height-padding);
  textSize(bigText);
  fill(black);
  text(stylizedRepCount, 50, height-padding/2);
  fill(lightGray);
  text("/ 5", 100, height-padding/2);
  
  /* Bottom right text*/
  textAlign(RIGHT);
  textSize(smallText);
  fill(lightGray);
  text("NEXT EXERCISE", width-padding/2, height-padding);
  textSize(bigText);
  fill(black);
  text(exercises[exCount+1], width-padding/2, height-padding/2);
  
  /* Central image */
  imageMode(CENTER);
  image(imgAGood, width/2, height*.43, imgWidth, imgHeight);
  }
  
  if (state == 11){          //        -----------------------------------------------------------   EXERCISE A: BAD FORM 1   --------------------------------------------------------------------------------------
  textAlign(CENTER);
  background(255);
  
  /* Top text */
  textSize(smallText);
  fill(lightGray);
  text("CURRENT EXERCISE", width/2, padding/2);
  textSize(bigText+bigText/5);
  fill(black);
  text(exercises[exCount], width/2, padding+padding/8);
  
  /* Bottom left text*/
  textAlign(LEFT);
  textSize(smallText);
  fill(lightGray);
  text("REP COUNT", 50, height-padding);
  textSize(bigText);
  fill(black);
  text(stylizedRepCount, 50, height-padding/2);
  fill(lightGray);
  text("/ 5", 100, height-padding/2);
  
  /* Bottom right text*/
  textAlign(RIGHT);
  textSize(smallText);
  fill(lightGray);
  text("NEXT EXERCISE", width-padding/2, height-padding);
  textSize(bigText);
  fill(black);
  text(exercises[exCount+1], width-padding/2, height-padding/2);
  
  /* Central image */
  imageMode(CENTER);
  image(imgABad, width/2, height*.43, imgWidth, imgHeight);
  fill(deepRed);
  textAlign(CENTER);
  textSize(3*bigText/4);
  text("Try to keep your arms further from your chest.", width/2, height/2+140);
  }
  
  if (state == 20){          //        -----------------------------------------------------------   EXERCISE B: GOOD FORM    --------------------------------------------------------------------------------------
  textAlign(CENTER);
  background(255);
  
  /* Top text */
  textSize(smallText);
  fill(lightGray);
  text("CURRENT EXERCISE", width/2, padding/2);
  textSize(bigText+bigText/5);
  fill(black);
  text(exercises[exCount+1], width/2, padding+padding/8);
  
  /* Bottom left text*/
  textAlign(LEFT);
  textSize(smallText);
  fill(lightGray);
  text("REP COUNT", 50, height-padding);
  textSize(bigText);
  fill(black);
  text(stylizedRepCount, 50, height-padding/2);
  fill(lightGray);
  text("/ 5", 100, height-padding/2);
  
  /* Bottom right text*/
  textAlign(RIGHT);
  textSize(smallText);
  fill(lightGray);
  text("NEXT EXERCISE", width-padding/2, height-padding);
  textSize(bigText);
  fill(black);
  text(exercises[exCount+2], width-padding/2, height-padding/2);
  
  /* Central image */
  imageMode(CENTER);
  image(imgBGood, width/2, height*.43, imgWidth, imgHeight);
  }
  if (state == 21){          //        -----------------------------------------------------------   EXERCISE B: BAD FORM 1   --------------------------------------------------------------------------------------
  textAlign(CENTER);
  background(255);
  
  /* Top text */
  textSize(smallText);
  fill(lightGray);
  text("CURRENT EXERCISE", width/2, padding/2);
  textSize(bigText+bigText/5);
  fill(black);
  text(exercises[exCount+1], width/2, padding+padding/8);
  
  /* Bottom left text*/
  textAlign(LEFT);
  textSize(smallText);
  fill(lightGray);
  text("REP COUNT", 50, height-padding);
  textSize(bigText);
  fill(black);
  text(stylizedRepCount, 50, height-padding/2);
  fill(lightGray);
  text("/ 5", 100, height-padding/2);
  
  /* Bottom right text*/
  textAlign(RIGHT);
  textSize(smallText);
  fill(lightGray);
  text("NEXT EXERCISE", width-padding/2, height-padding);
  textSize(bigText);
  fill(black);
  text(exercises[exCount+2], width-padding/2, height-padding/2);
  
  /* Central image */
  imageMode(CENTER);
  image(imgBBad, width/2, height*.43, imgWidth, imgHeight);
  fill(deepRed);
  textAlign(CENTER);
  textSize(3*bigText/4);
  text("Try to raise your arms up higher.", width/2, height/2+140);
  }
  
  if (state == 30){          //        -----------------------------------------------------------   EXERCISE C: GOOD FORM   --------------------------------------------------------------------------------------
    textAlign(CENTER);
  background(255);
  
  /* Top text */
  textSize(smallText);
  fill(lightGray);
  text("CURRENT EXERCISE", width/2, padding/2);
  textSize(bigText+bigText/5);
  fill(black);
  text(exercises[exCount+2], width/2, padding+padding/8);
  
  /* Bottom left text*/
  textAlign(LEFT);
  textSize(smallText);
  fill(lightGray);
  text("REP COUNT", 50, height-padding);
  textSize(bigText);
  fill(black);
  text(stylizedRepCount, 50, height-padding/2);
  fill(lightGray);
  text("/ 5", 100, height-padding/2);
  
  /* Bottom right text*/
  textAlign(RIGHT);
  textSize(smallText);
  fill(lightGray);
  text("NEXT EXERCISE", width-padding/2, height-padding);
  textSize(bigText);
  fill(black);
  text(exercises[exCount], width-padding/2, height-padding/2);
  
  /* Central image */
  imageMode(CENTER);
  image(imgCGood, width/2, height*.43-15, imgWidth, imgHeight+30);
  }
  if (state == 31){          //        -----------------------------------------------------------   EXERCISE C: BAD FORM 1   --------------------------------------------------------------------------------------
  textAlign(CENTER);
  background(255);
  
  /* Top text */
  textSize(smallText);
  fill(lightGray);
  text("CURRENT EXERCISE", width/2, padding/2);
  textSize(bigText+bigText/5);
  fill(black);
  text(exercises[exCount+2], width/2, padding+padding/8);
  
  /* Bottom left text*/
  textAlign(LEFT);
  textSize(smallText);
  fill(lightGray);
  text("REP COUNT", 50, height-padding);
  textSize(bigText);
  fill(black);
  text(stylizedRepCount, 50, height-padding/2);
  fill(lightGray);
  text("/ 5", 100, height-padding/2);
  
  /* Bottom right text*/
  textAlign(RIGHT);
  textSize(smallText);
  fill(lightGray);
  text("NEXT EXERCISE", width-padding/2, height-padding);
  textSize(bigText);
  fill(black);
  text(exercises[exCount], width-padding/2, height-padding/2);
  
  /* Central image */
  imageMode(CENTER);
  image(imgCBad, width/2, height*.43-15, imgWidth, imgHeight+30);
  fill(deepRed);
  textAlign(CENTER);
  textSize(3*bigText/4);
  text("Don't forget to rotate your wrists.", width/2, height/2+140);
  }

}





void keyPressed() {
  if (key == ENTER) {
    if (tCnt>0 || type>0) {
      if (!firstTrained) firstTrained = true;
      resetSVM();
    } else {
      println("Error: No Data");
    }
  }
  if (key >= '0' && key <= '9') {
    type = key - '0';
  }
  if (key == TAB) {
    if (tCnt>0) { 
      if (type<(colors.length-1))++type;
      tCnt = 0;
    }
  }
  if (key == '/') {
    firstTrained = false;
    resetSVM();
    clearSVM();
  }
  if (key == 'S' || key == 's') {
    if (model!=null) { 
      saveSVM_Model(sketchPath()+"/data/test.model", model);
      println("Model Saved");
    }
  }
  if (key == ' ') {
    if (b_pause == true) b_pause = false;
    else b_pause = true;
  }
  if (key == 'A' || key == 'a') {
    activationThld = min(activationThld+10, 100);
  }
  if (key == 'Z' || key == 'z') {
    activationThld = max(activationThld-10, 10);
  }
}

//Tool functions

//Append a value to a float[] array.
float[] appendArray (float[] _array, float _val) {
  float[] array = _array;
  float[] tempArray = new float[_array.length-1];
  arrayCopy(array, tempArray, tempArray.length);
  array[0] = _val;
  arrayCopy(tempArray, 0, array, 1, tempArray.length);
  return array;
}

////Draw a line graph to visualize the sensor stream
//void lineGraph(float[] data, float _l, float _u, float _x, float _y, float _w, float _h, int _index) {
//  color colors[] = {
//    color(255, 0, 0), color(0, 255, 0), color(0, 0, 255), color(255, 255, 0), color(0, 255, 255), 
//    color(255, 0, 255), color(0)
//  };
//  int index = min(max(_index, 0), colors.length);
//  pushStyle();
//  float delta = _w/data.length;
//  beginShape();
//  noFill();
//  stroke(colors[index]);
//  for (float i : data) {
//    float h = map(i, _l, _u, 0, _h);
//    vertex(_x, _y+h);
//    _x = _x + delta;
//  }
//  endShape();
//  popStyle();
//}

////Draw a bar graph to visualize the modeArray
//void barGraph(float[] data, float _l, float _u, float _x, float _y, float _w, float _h) {
//  color colors[] = {
//    color(155, 89, 182), color(63, 195, 128), color(214, 69, 65), color(82, 179, 217), color(244, 208, 63), 
//    color(242, 121, 53), color(0, 121, 53), color(128, 128, 0), color(52, 0, 128), color(128, 52, 0)
//  };
//  pushStyle();
//  noStroke();
//  float delta = _w / data.length;
//  for (int p = 0; p < data.length; p++) {
//    float i = data[p];
//    int cIndex = min((int) i, colors.length-1);
//    if (i<0) fill(255, 100);
//    else fill(colors[cIndex], 100);
//    float h = map(_u, _l, _u, 0, _h);
//    rect(_x, _y-h, delta, h);
//    _x = _x + delta;
//  }
//  popStyle();
//}