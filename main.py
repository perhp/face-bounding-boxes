import cv2
from mtcnn import MTCNN

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    detector = MTCNN()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = detector.detect_faces(frame_rgb)

        for face in detections:
            x, y, w, h = face['box']
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            keypoints = face['keypoints']
            for key, point in keypoints.items():
                cv2.circle(frame, point, 2, (0, 0, 255), 2)

        cv2.imshow('Webcam Face Detection (MTCNN)', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
