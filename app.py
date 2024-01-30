from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (256, 256))
    return resized_frame

def predict_image(model, image):
    
    image = np.expand_dims(image, axis=0)
    
    predictions = model.predict(image)
    
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    class_probabilities = predictions[0]
    
    return predicted_class, class_probabilities


def generate_frames():
    from tensorflow.keras.models import load_model

    class_labels = {
        0: 'Ayrshire cattle',
        1: 'Holstein Friesian cattle',
        2: 'Brown Swiss cattle',
        3: 'Red Dane cattle',
        4: 'Jersey cattle'}

    model = load_model('models/cattle_detector.h5')

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        
        processed_frame = preprocess_frame(frame)

        predicted_class, percentages = predict_image(model, processed_frame/255)

        perc = percentages.max()

        if perc > 0.85:
            cv2.putText(frame, f'{class_labels[predicted_class]} : {perc*100}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f'Not a cow', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        if not success:
            break

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False,host="0.0.0.0")

