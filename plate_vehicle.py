import yolov5
from PIL import Image
from hezar.models import Model
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

class PlateVehicle:
    def __init__(self, yolov5_model_path, crnn_model_path, trocr_processor_path, trocr_model_path):
        self.yolov5_model = yolov5.load(yolov5_model_path)
        self.trocr_processor = TrOCRProcessor.from_pretrained(trocr_processor_path)
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(trocr_model_path)
        self.configure_yolov5()

    def configure_yolov5(self):
        self.yolov5_model.conf = 0.25  # Umbral de confianza para NMS
        self.yolov5_model.iou = 0.45  # Umbral IoU para NMS
        self.yolov5_model.agnostic = False  # No agnóstico respecto a las clases
        self.yolov5_model.multi_label = False  # No permitir múltiples etiquetas por caja
        self.yolov5_model.max_det = 1000  # Máximo número de detecciones por imagen

    def load_image(self, img_path):
        return Image.open(img_path)

    def detect_plates(self, img, augment=False):
        results = self.yolov5_model(img, size=640, augment=augment)
        predictions = results.pred[0]
        boxes = predictions[:, :4]  # Coordenadas de las cajas (x1, y1, x2, y2)
        boxes = boxes.int().numpy()  # Convertir las coordenadas a enteros y numpy
        return boxes

    def perform_ocr(self, cropped_img):
        image = cropped_img.convert("RGB")
        pixel_values = self.trocr_processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.trocr_model.generate(pixel_values)
        generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def clean_text(self, text):
        # Eliminar caracteres no alfanuméricos
        return re.sub(r'[^a-zA-Z0-9]', '', text)

    def process_plate(self, box, img):
        x1, y1, x2, y2 = box
        cropped_img = img.crop((int(x1), int(y1), int(x2), int(y2)))

        # Perform OCR on the cropped image
        ocr_text = self.perform_ocr(cropped_img)
        clean_text = self.clean_text(ocr_text)
        return clean_text

    def process_image(self, img_path):
        img = self.load_image(img_path)
        boxes = self.detect_plates(img)
        ocr_texts = []

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_plate, box, img) for box in boxes]

            for future in as_completed(futures):
                ocr_texts.append(future.result())

        # Concatenar todos los textos limpios en un solo string plano
        flat_text = ' '.join(ocr_texts)
        return flat_text

    def process_images(self, img_paths):
        ocr_texts = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_image, img_path) for img_path in img_paths]

            for future in as_completed(futures):
                ocr_texts.append(future.result())

        return ocr_texts
