# Face Anti-Spoofing and Verification Demo

This project is a Gradio-based application that demonstrates facial verification, attribute analysis, and anti-spoofing using the DeepFace library. The application uses multiple models to detect and verify faces, analyze facial attributes, and ensure that the detected face is real.

## Features

- **Face Verification:** Compare two images to check if they belong to the same person using various detection backends.
- **Facial Attribute Analysis:** Analyze attributes like age, gender, race, and emotion.
- **Anti-Spoofing Check:** Ensure that the detected face is not a spoof using multiple models.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/face-anti-spoofing-demo.git
   ```
2. Navigate to the project directory:
   ```bash
   cd face-anti-spoofing-demo
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Gradio application using the following command:
   ```bash
   gradio deepface_demo.py --demo-name=iface
   ```
2. Open the provided URL in your web browser.
3. Upload images to verify faces, analyze facial attributes, and perform anti-spoofing checks.

## Models Used

- **Face Verification and Anti-Spoofing:** Facenet512, VGG-Face, DeepFace, ArcFace
- **Facial Attribute Analysis:** Age, Gender, Race, Emotion analysis using DeepFace

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DeepFace Library](https://github.com/serengil/deepface)
- [Gradio](https://gradio.app/)

This `README.md` is now formatted entirely in Markdown, providing a clear guide for setting up and running the Gradio-based Face Anti-Spoofing and Verification Demo.
