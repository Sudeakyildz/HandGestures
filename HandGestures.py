import cv2
import mediapipe
import pyttsx3
import time

class ElHareketi:
    def __init__(self):
        # Initialize the camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Camera could not be opened!")

        # Initialize MediaPipe Hands module
        self.mpHands = mediapipe.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mediapipe.solutions.drawing_utils
        self.checkThumbsUp = False  # Flag to check thumbs up gesture

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)    # Speed of the speech
        self.engine.setProperty('volume', 1.0)  # Volume level

    def run(self):
        try:
            while True:
                # Read frame from the camera
                success, img = self.camera.read()
                if not success:
                    print("Could not read frame from camera!")
                    break

                # Convert image to RGB (MediaPipe uses RGB)
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                hlms = self.hands.process(imgRGB)  # Detect hands
                height, width, _ = img.shape

                if hlms.multi_hand_landmarks:
                    for handlandmarks in hlms.multi_hand_landmarks:
                        for fingerNum, landmark in enumerate(handlandmarks.landmark):
                            positionX, positionY = int(landmark.x * width), int(landmark.y * height)

                            # If any finger (except thumb) is raised, break
                            if fingerNum > 4 and landmark.y < handlandmarks.landmark[2].y:
                                break

                            # Check if thumb is up
                            if fingerNum == 20 and landmark.y > handlandmarks.landmark[2].y:
                                self.checkThumbsUp = True

                        # Draw landmarks on the hand
                        self.mpDraw.draw_landmarks(img, handlandmarks, self.mpHands.HAND_CONNECTIONS)

                if self.checkThumbsUp:
                    print("Thumbs up detected!")

                    # Voice greeting
                    self.engine.say("Welcome!")
                    self.engine.runAndWait()

                    # Draw "Welcome" text on screen
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 2
                    thickness = 5
                    text = "Hos geldiniz"
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    x = (width - text_width) // 2
                    y = (height + text_height) // 2

                    cv2.putText(img, text, (x, y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
                    cv2.imshow("Camera", img)
                    cv2.waitKey(1)

                    # Wait for 3 seconds before exiting
                    time.sleep(3)
                    break

                # Show camera frame
                cv2.imshow("Camera", img)

                # Press 'q' to quit the program
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            # Release camera and close windows
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    el_hareketi = ElHareketi()
    el_hareketi.run()

