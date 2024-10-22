# Golf Swing Comparative Feedback System

This project aims to create an affordable and accessible comparison tool for golfers by utilizing computer vision. With just a phone camera, users can record their swings and compare them to those of professional golfers. The goal is to provide golfers with actionable feedback to help refine their technique without the need for expensive specialist equipment.

## Key Features
Swing Comparison: Record a video of your swing and compare it with a professional golfer's swing.
Event Detection: Utilizes the 'SwingNet' model to identify eight key events in both your swing and the reference swing, ensuring synchronization.
Pose Estimation: Uses MediaPipe for real-time pose detection to extract key body part coordinates.
Normalization: Normalizes pose data to account for differences in image size, golfer height, and distance from the camera.
Swing Similarity: Uses Euclidean distance to measure the similarity between your pose and the professional's pose at key swing events.
Feedback: Highlights areas where your swing deviates most from the professional’s, allowing you to focus on improving specific aspects.
Visual Comparison: Generates images that allow easy comparison of your swing against the reference golfer at synchronized points.
GUI: An intuitive interface allows users to upload videos, visualize their swing analysis, and receive tailored feedback.

## GUI output - Swing suggestions
![gui](https://github.com/user-attachments/assets/879154e1-cc47-4c3c-aa21-6517e2458ca5)

## Image output - Swing comparison using mediapipe pose estimation
![image](https://github.com/user-attachments/assets/481b924c-697f-47a5-9b67-388b276ee08f)
