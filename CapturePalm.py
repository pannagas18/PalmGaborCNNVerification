import cv2
import numpy as np

class CapturePalm:
    def __init__(self) -> None:
        pass

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Captures two images, one for registration and the other for verification.
        """
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Video Feed")

        outline_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        rect_x, rect_y, rect_w, rect_h = 200, 150, 150, 150 # 200, 150, 240, 250

        cv2.line(outline_image, (rect_x,rect_y), (rect_x + rect_w,rect_y), (255, 255, 255), 2, cv2.LINE_AA)

        cv2.circle(outline_image, (rect_x,rect_y), 5,(0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(outline_image, (rect_x + rect_w,rect_y), 5,(0, 0, 255), -1, cv2.LINE_AA)
        rect_x, rect_y, rect_w, rect_h = 180, 180, 190, 210
        cv2.rectangle(outline_image, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2, cv2.LINE_AA)
        
        print("Align the two points with the ring-little finger andthe index-middle finger valley points")
        text_message1 = "ring-little valley"
        text_message2 = "index-middle valley"

        input1 = None
        input2 = None

        while True:
            _, frame = cap.read()

            composite_image = cv2.addWeighted(frame, 1, outline_image, 1, 0)

            cv2.putText(composite_image, text_message1, (rect_x-100, rect_y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(composite_image, text_message2, (rect_x + rect_w,rect_y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Video Feed", composite_image)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            key = cv2.waitKey(1)

            if key == ord('s') and input1 is None:

                input1 = frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w].copy()
                print("First palm image captured successfully!")

                cv2.destroyWindow("Video Feed")

            elif key == ord('s') and input1 is not None and input2 is None:

                input2 = frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w].copy()

                print("Second palm image saved successfully!")

                cv2.destroyWindow("Video Feed")

                break

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        cv2.imwrite("input1.png", input1)
        cv2.imwrite("input2.png", input2)
        
        return input1, input2
