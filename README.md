# Dental AI

Dental AI is an AI-powered web application designed to analyze dental X-ray images and medical reports. It uses YOLOv8 for object detection and Google's Gemini AI for generating diagnoses based on detected issues and extracted report text.

## Features

- **X-ray Image Analysis:** Detects dental issues from uploaded X-ray images using YOLOv8.
- **AI-Powered Diagnosis:** Generates a diagnosis based on detected issues using Gemini AI.
- **Medical Report Analysis:** Extracts text from uploaded PDF reports and provides insights.
- **Web-based Interface:** Simple and easy-to-use Flask web app with real-time processing.

## Installation

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/R-Vaishnav-Raj/dental-ai.git
   cd dental-ai
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python object_detector.py
   ```

The app will be accessible at `http://127.0.0.1:8080/`.

## API Endpoints

### 1. `/detect` (POST)

Uploads an X-ray image and detects dental issues.

### 2. `/diagnose` (POST)

Generates a diagnosis based on the detected issues.

### 3. `/upload_report` (POST)

Uploads a medical report (PDF) and extracts text for AI analysis.

![Demo Video](dentalaidemo.mp4)


## Acknowledgments

Special thanks to [SubGlitch1/DentalXrayAI](https://github.com/SubGlitch1/DentalXrayAI) for guidance on training the AI model for scanning dental X-ray images.

##

