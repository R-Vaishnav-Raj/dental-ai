from flask import request, Flask, jsonify, render_template
from waitress import serve
import google.generativeai as genai
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import pdfplumber  # For extracting text from PDFs

app = Flask(__name__)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyCABaPaHrEp-0LmYVb5gSE2SSUyY1h6DX8"  # Replace with a valid API key
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')

# Initialize YOLOv8 model
yolo_model = YOLO("best.pt")  # Replace with your YOLOv8 model path

# Store last detections for diagnosis
last_detections = []


@app.route("/")
def root():
    """
    Serve the main page.
    """
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    """
    Detect issues from the uploaded image using YOLOv8 and store them.
    """
    global last_detections

    # Check if an image file is uploaded
    if "image_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # Open the uploaded image
    buf = request.files["image_file"]
    image = Image.open(buf)

    # Detect objects using YOLOv8 and draw bounding boxes
    detections, annotated_image = detect_objects_on_image(image)
    last_detections = detections  # Store detections for later diagnosis

    # Save the annotated image to a temporary file
    annotated_image_path = "static/annotated_image.jpg"  # Save in the static folder for frontend access
    annotated_image.save(annotated_image_path)

    return jsonify({
        "detections": detections,
        "annotated_image": annotated_image_path
    })


@app.route("/diagnose", methods=["POST"])
def diagnose():
    """
    Generate a diagnosis based on the last detected issues using Gemini AI.
    """
    global last_detections
    if not last_detections:
        return jsonify({"diagnosis": "No detections available. Please upload an image first."})

    diagnosis = get_gemini_diagnosis(last_detections)
    return jsonify({"diagnosis": diagnosis})


@app.route("/upload_report", methods=["POST"])
def upload_report():
    """
    Extract text from the uploaded PDF medical report and generate a diagnosis.
    """
    if "pdf_file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["pdf_file"]
    extracted_text = extract_text_from_pdf(pdf_file)

    if not extracted_text.strip():
        return jsonify({"error": "Failed to extract text from PDF"}), 400

    diagnosis = get_gemini_diagnosis_from_report(extracted_text)
    return jsonify({"diagnosis": diagnosis})


def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.
    """
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    return text

def detect_objects_on_image(image):
    """
    Detect objects in the image using YOLOv8 and draw bounding boxes.
    """
    results = yolo_model.predict(image)
    result = results[0]
    output = []

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 16)  # Load a proper font
    except IOError:
        font = ImageFont.load_default()  # Fallback to default if unavailable

    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        prob_percentage = f"{prob * 100:.2f}%"
        output.append([x1, y1, x2, y2, result.names[class_id], prob_percentage])

        # **Green bounding box** using a valid format
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

        # Text background for readability
        text = f"{result.names[class_id]} {prob_percentage}"
        text_size = draw.textbbox((0, 0), text, font=font)  
        text_width, text_height = text_size[2] - text_size[0], text_size[3] - text_size[1]

        draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="green")  
        draw.text((x1, y1 - text_height), text, fill="black", font=font)  # Black text

    return output, image


def get_gemini_diagnosis(detections):
    """
    Generate a diagnosis from the detected issues using Gemini AI.
    """
    formatted_detections = [
        f"{detection[4]} (Confidence: {detection[5]})" for detection in detections
    ]

    prompt = (
        "You are a dental X-ray diagnosis assistant. Based on the following detected issues in an X-ray image, "
        "provide a diagnosis and possible dental conditions:\n\n"
        f"Detected Issues:\n" + "\n".join(formatted_detections) + "\n\n"
        "1. Headings for required treatments ordered by urgency.\n"
        "2. No introductory statements; respond concisely."
    )

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Failed to generate diagnosis: {str(e)}"


def get_gemini_diagnosis_from_report(text):
    """
    Generate a diagnosis from the extracted medical report using Gemini AI.
    """
    prompt = (
        "You are an expert dentist analyzing patient medical reports. Based on the following medical data, "
        "provide a possible diagnosis and conditions the patient may have:\n\n"
        f"Medical Report:\n{text}\n\n"
        "1. Highlight potential health issues based on symptoms.\n"
        "2. Provide insights on possible diseases or conditions.\n"
        "3. Do not include warnings; this report is reviewed by a doctor."
    )

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Failed to generate diagnosis: {str(e)}"


serve(app, host='0.0.0.0', port=8080)