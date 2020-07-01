An eye-tracker that uses computer vision and machine learning to determine where you are looking at,
while also being able to keep up with head and eye movements. 

The main program is "EyeTrackerFinal.py," and the machine-learning functions are in "LineOfBestFit.py." 

Previous iterations can be found in "extras" folder, but they are uncommented and do not operate fully. 

Calibration images will be saved into the "analysis" folder.

Instructions:
*  Download the shape_predictor_68_face_landmarks.dat file from the shape_predictor_68_face_landmarks.txt file link
*  On the calibration screen, press the 'c' key, while looking at the colored dot on the screen.
*  Press (Hold) 'w' to skip the calibration sequence (only works if you already have taken calibration images already).
*  Press (Hold) 'e' to toggle a static border for the bounds of the eye detection. Better to leave it untoggled, but
it is nice to have it just in case (mainly for debugging).
*  Press (Hold) 'q' to quit the program.

After you perform the calibration progress, it will take input from your screen, and place a dot where the program thinks 
you are looking at. It is not the most accurately, but the code should be making it as accurate as possible. Currently
looking into having the dot become a normal overlay on the screen, instead of the screen being in a separate window.

Will be updating whenver I have free time.
