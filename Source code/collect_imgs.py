import os
import cv2

# Define directory dynamically relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')

# Create the directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 37 #0-9,A-Z,Next
dataset_size = 100

cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

# Loop through each class
for j in range(number_of_classes):
    # Create a directory for the current class
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))


    # Display a message to prompt the user to start capturing images
    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the prompt message on the frame    
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Wait for the user to press the 'q' key to start capturing images
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0

    # Capture images until the desired dataset size is reached
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break


        # Display the frame    
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save the captured frame as an image in the class directory
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        counter += 1

# release video capture object and Close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
