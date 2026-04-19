# Lab 5: TensorRT RetinaNet

Простая реализация детекции объектов на видео с помощью RetinaNet и TensorRT.

На вход подается видео (`mp4`, `avi` и т.д.),  
на выходе получается видео с рамками, классами и confidence.

## Что сделано

- Экспорт RetinaNet в ONNX
- Сборка TensorRT engine (INT8 по умолчанию, есть FP16 и FP32)
- Обработка видео по кадрам
- Декодирование предсказаний, NMS и фильтрация по confidence
- Сохранение результата в выходной видеофайл

Проект рабочий: сборка engine и инференс запускаются корректно.

## Тестовый стенд

- GPU: **NVIDIA GeForce RTX 4070 Laptop GPU**
- TensorRT: **10.3.0**

## Установка

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install tensorrt==10.3.0 tensorrt-cu12-bindings==10.3.0 tensorrt-cu12-libs==10.3.0
pip install opencv-python onnx numpy
```

## Структура проекта

- `src/build_engine.py` - сборка ONNX и TensorRT engine
- `src/main.py` - запуск обработки видео
- `src/retinanet_trt/` - основная логика проекта
  - `config.py` - настройки
  - `engine_builder.py` - экспорт и сборка engine
  - `trt_runtime.py` - инференс через TensorRT
  - `postprocess.py` - декодирование и NMS
  - `calibration.py` - INT8 калибровка по кадрам видео
  - `video_utils.py` - подготовка кадров и отрисовка
- `data/videos/` - входные/выходные видео
- `artifacts/` - ONNX, TensorRT engine и кэш калибрации

## Как запустить

1. Собрать engine (INT8 по умолчанию):

```bash
python src/build_engine.py
```

INT8 калибровка делается автоматически по кадрам из `data/videos/test.mp4`.

Если нужно указать своё видео для калибровки:

```bash
python src/build_engine.py --precision int8 --calib-video data/videos/test.mp4 --calib-frames 128
```

Если нужен FP16:

```bash
python src/build_engine.py --precision fp16
```

Если нужен FP32:

```bash
python src/build_engine.py --precision fp32
```

2. Запустить обработку:

```bash
python src/main.py
```

По умолчанию:
- вход: `data/videos/test.mp4`
- выход: `data/videos/output.mp4`

Пример с параметрами:

```bash
python src/main.py --input data/videos/test.mp4 --output data/videos/output.mp4 --conf 0.5
```

Короткий тест на первых N кадрах:

```bash
python src/main.py --max-frames 100
```

В конце запуска выводится:
- средний FPS обработки
- `Speed vs real-time: Xx`
- статус: быстрее или медленнее реального времени
