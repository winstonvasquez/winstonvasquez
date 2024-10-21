from fastapi import FastAPI, File, UploadFile, HTTPException
from plate_vehicle import PlateVehicle
from typing import List
import shutil
import os

app = FastAPI()

class VehicleRest:
    def __init__(self, plate_vehicle):
        self.plate_vehicle = plate_vehicle

    async def process_image(self, files: List[UploadFile]):
        # Verificar que se haya enviado un archivo
        if not files:
            raise HTTPException(status_code=400, detail="No file received")

        # Obtener el primer archivo recibido
        file = files[0]

        # Guardar la imagen recibida en el servidor
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Procesar la imagen y obtener el texto plano
        flat_text = self.plate_vehicle.process_image(temp_file_path)

        # Eliminar el archivo temporal
        os.remove(temp_file_path)

        return flat_text

# Inicializar vehicle_rest
yolov5_model_path = "keremberke/yolov5m-license-plate"
crnn_model_path = "hezarai/crnn-fa-64x256-license-plate-recognition"
trocr_processor_path = "microsoft/trocr-base-printed"
trocr_model_path = "microsoft/trocr-base-printed"

# Crear instancia de PlateVehicle
plate_vehicle = PlateVehicle(
    yolov5_model_path, crnn_model_path, trocr_processor_path, trocr_model_path
)

# Crear instancia de VehicleRest
vehicle_rest = VehicleRest(plate_vehicle)

@app.post("/process_image")
async def process_image(files: List[UploadFile] = File(...)):

    flat_text = await vehicle_rest.process_image(files)
    return {"plate": flat_text}

if __name__ == "__main__":
    import uvicorn
    # Ejecutar el servidor FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)
