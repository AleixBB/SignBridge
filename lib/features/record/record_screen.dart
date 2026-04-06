import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart'; //acceder a la camara del mobil
import 'package:image/image.dart' as img; //procesar imagenes (convertir a jpg)
import 'package:http/http.dart' as http; //enviar frames al servidor
import 'dart:io' show Platform; //saber si es android o ios
import 'dart:convert';
import 'dart:async';

class RecordScreen extends StatefulWidget {
  @override
  _RecordScreenState createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> with SingleTickerProviderStateMixin {
  CameraController? _controller; //controla la camera
  bool _isProcessing = false;
  bool _isInitializing = true; //indica si está cargando
  String? _errorMessage; //guarda errores

  // Variables para la detección
  String _detectedLetter = '-';
  double _confidence = 0.0;
  String _currentWord = '';
  List<String> _detectedLetters = [];
  bool _hasHand = false; //indica si hay una mano visible

  // Animación para cuando se detecta una letra
  late AnimationController _animationController;
  late Animation<double> _scaleAnimation;
  bool _showLetterAnimation = false;

  // Control para estabilidad
  Timer? _wordBuilderTimer; //evita que se añadan letras demasiado rápido
  String _lastLetter = '-';

  @override
  void initState() { //inicializa la camara y configura animación
    super.initState();
    _initializeCamera();

    _animationController = AnimationController(
      duration: const Duration(milliseconds: 300),
      vsync: this,
    );
    _scaleAnimation = Tween<double>(begin: 1.0, end: 1.2).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.elasticOut),
    );
  }

  @override
  void dispose() { //libera recursos
    _wordBuilderTimer?.cancel();
    _animationController.dispose();
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _initializeCamera() async { //inicializar camara
    try {
      final cameras = await availableCameras();
      //obtiene cameras disponibles
      if (cameras.isEmpty) {
        setState(() {
          _errorMessage = 'No camera available on this device';
          _isInitializing = false;
        });
        return;
      }

      final frontCamera = cameras.firstWhere(
            (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => cameras[0],
      );

      _controller = CameraController(
        frontCamera,
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await _controller!.initialize();

      setState(() {
        _isInitializing = false;
      });
      //empieza stream de imagenes
      await _controller!.startImageStream((image) {
        if (!_isProcessing) {
          _isProcessing = true;
          _processFrame(image);
        }
      });

    } catch (e) {
      setState(() {
        _errorMessage = 'Error initializing camera: $e';
        _isInitializing = false;
      });
    }
  }
  //como la camara puede darnos imagenes en diferentes formatos convertimos todas a Jpeg
  Future<Uint8List?> _convertCameraImageToJpeg(CameraImage image) async {
    try {
      if (Platform.isIOS && image.planes.length == 1) {
        return await _convertIOSImageToJpeg(image);
      }

      if (Platform.isAndroid && image.planes.length >= 3) {
        return await _convertYUV420ToJpeg(image);
      }

      if (image.planes.length == 2) {
        return await _convertNV21ToJpeg(image);
      }

      if (image.planes.length == 1) {
        return await _convertSinglePlaneToJpeg(image);
      }

      return null;
    } catch (e) {
      return null;
    }
  }

  Future<Uint8List?> _convertIOSImageToJpeg(CameraImage image) async {
    final width = image.width;
    final height = image.height;
    final plane = image.planes[0];
    final bytes = plane.bytes;

    final imgImage = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixelIndex = (y * plane.bytesPerRow) + (x * 4);

        if (pixelIndex + 3 < bytes.length) {
          final b = bytes[pixelIndex];
          final g = bytes[pixelIndex + 1];
          final r = bytes[pixelIndex + 2];

          imgImage.setPixelRgb(x, y, r, g, b);
        }
      }
    }

    return Uint8List.fromList(img.encodeJpg(imgImage, quality: 70));
  }

  Future<Uint8List?> _convertYUV420ToJpeg(CameraImage image) async {
    final width = image.width;
    final height = image.height;
    final imgImage = img.Image(width: width, height: height);

    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final yIndex = y * yPlane.bytesPerRow + x;
        final uvIndex = (y ~/ 2) * uPlane.bytesPerRow + (x ~/ 2);

        final Y = yPlane.bytes[yIndex];
        final U = uPlane.bytes[uvIndex];
        final V = vPlane.bytes[uvIndex];

        int r = (Y + 1.403 * (V - 128)).round();
        int g = (Y - 0.344 * (U - 128) - 0.714 * (V - 128)).round();
        int b = (Y + 1.770 * (U - 128)).round();

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        imgImage.setPixelRgb(x, y, r, g, b);
      }
    }

    return Uint8List.fromList(img.encodeJpg(imgImage, quality: 70));
  }

  Future<Uint8List?> _convertNV21ToJpeg(CameraImage image) async {
    final width = image.width;
    final height = image.height;
    final imgImage = img.Image(width: width, height: height);

    final yPlane = image.planes[0];
    final uvPlane = image.planes[1];

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final yIndex = y * yPlane.bytesPerRow + x;
        final uvIndex = (y ~/ 2) * uvPlane.bytesPerRow + (x ~/ 2) * 2;

        final Y = yPlane.bytes[yIndex];
        final U = uvPlane.bytes[uvIndex];
        final V = uvPlane.bytes[uvIndex + 1];

        int r = (Y + 1.403 * (V - 128)).round();
        int g = (Y - 0.344 * (U - 128) - 0.714 * (V - 128)).round();
        int b = (Y + 1.770 * (U - 128)).round();

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        imgImage.setPixelRgb(x, y, r, g, b);
      }
    }

    return Uint8List.fromList(img.encodeJpg(imgImage, quality: 70));
  }

  Future<Uint8List?> _convertSinglePlaneToJpeg(CameraImage image) async {
    final width = image.width;
    final height = image.height;
    final plane = image.planes[0];

    final imgImage = img.Image(width: width, height: height);
    final bytesPerPixel = plane.bytesPerRow ~/ width;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixelIndex = (y * plane.bytesPerRow) + (x * bytesPerPixel);

        if (pixelIndex + 2 < plane.bytes.length) {
          final b = plane.bytes[pixelIndex];
          final g = plane.bytes[pixelIndex + 1];
          final r = plane.bytes[pixelIndex + 2];

          imgImage.setPixelRgb(x, y, r, g, b);
        }
      }
    }

    return Uint8List.fromList(img.encodeJpg(imgImage, quality: 70));
  }

  //procesar frame
  Future<void> _processFrame(CameraImage image) async {
    try {
      //convertir img
      final jpeg = await _convertCameraImageToJpeg(image);

      if (jpeg == null) {
        _isProcessing = false;
        return;
      }
      //enviar al servidor
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('http://192.168.0.34:8000/process_frame'),
      );

      request.files.add(
        http.MultipartFile.fromBytes(
          'file',
          jpeg,
          filename: 'frame.jpg',
        ),
      );
      //recibir respuesta
      var response = await request.send().timeout(
        const Duration(seconds: 3),
        onTimeout: () {
          throw Exception('Request timeout');
        },
      );

      if (response.statusCode == 200) {
        var body = await response.stream.bytesToString();
        final data = json.decode(body);

        final newLetter = data['letter'] ?? '-';
        final newConfidence = (data['confidence'] ?? 0.0).toDouble();
        final letterChanged = data['letter_changed'] ?? false;
        final hasHand = data['has_hand'] ?? false;

        setState(() {
          _hasHand = hasHand;
          //si se cambia la letra ejecuta animacion y actualiza o añade letra
          if (letterChanged && newLetter != '-' && newConfidence > 60) {
            _detectedLetter = newLetter;
            _confidence = newConfidence;

            _showLetterAnimation = true;
            _animationController.forward().then((_) {
              _animationController.reverse();
              Future.delayed(Duration(milliseconds: 500), () {
                if (mounted) setState(() => _showLetterAnimation = false);
              });
            });

            if (_lastLetter != newLetter) {
              _lastLetter = newLetter;

              if (_wordBuilderTimer != null) _wordBuilderTimer!.cancel();
              _wordBuilderTimer = Timer(Duration(milliseconds: 500), () {
                if (mounted) {
                  setState(() {
                    _detectedLetters.add(newLetter);
                    if (_detectedLetters.length > 20) {
                      _detectedLetters.removeAt(0);
                    }
                    _currentWord = _detectedLetters.join('');
                  });
                }
              });
            }
          }
        });
      }
    } catch (e) {
      // Silent fail
    } finally {
      _isProcessing = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_isInitializing) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    if (_errorMessage != null) {
      return Scaffold(
        appBar: AppBar(title: const Text("SignBridge Camera")),
        body: Center(
          child: Text(_errorMessage!, style: const TextStyle(color: Colors.red)),
        ),
      );
    }

    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      body: Column(
        children: [
          // 📷 CÁMARA (gran parte de la pantalla)
          Expanded(
            flex: 6,
            child: Stack(
              children: [
                SizedBox(
                  width: double.infinity,
                  child: CameraPreview(_controller!),
                ),

                // Indicador encima de la cámara
                Positioned(
                  top: 50,
                  left: 20,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: Colors.black54,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      _hasHand ? "🟢 Mano detectada" : "🔍 Busca tu mano",
                      style: const TextStyle(color: Colors.white),
                    ),
                  ),
                ),
              ],
            ),
          ),

          // ⬜ PANEL BLANCO (traducción)
          Expanded(
            flex: 2,
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 15),
              decoration: const BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.only(
                  topLeft: Radius.circular(20),
                  topRight: Radius.circular(20),
                ),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // PALABRA
                  Text(
                    _currentWord.isEmpty ? '...' : _currentWord,
                    style: const TextStyle(
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                      color: Colors.black,
                    ),
                    textAlign: TextAlign.center,
                  ),

                  const SizedBox(height: 10),

                  // 🔠 LETRA ACTUAL
                  Text(
                    'Letra: ${_detectedLetter.toUpperCase()} (${_confidence.toStringAsFixed(1)}%)',
                    style: TextStyle(
                      fontSize: 14,
                      color: _confidence > 70 ? Colors.green : Colors.orange,
                    ),
                  ),

                  const SizedBox(height: 15),

                  // 🔘 BOTONES
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton(
                          onPressed: () {
                            setState(() {
                              _detectedLetters.clear();
                              _currentWord = '';
                              _lastLetter = '-';
                            });
                          },
                          child: const Text('Limpiar'),
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: ElevatedButton(
                          onPressed: () {
                            if (_currentWord.isNotEmpty) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(
                                  content: Text('Guardado: $_currentWord'),
                                  backgroundColor: Colors.green,
                                ),
                              );
                            }
                          },
                          child: const Text('Guardar'),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}