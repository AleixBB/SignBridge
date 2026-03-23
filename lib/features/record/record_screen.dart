import 'package:flutter/material.dart';
import '../../core/widgets/primary_button.dart';

class RecordScreen extends StatelessWidget {
  static const String routeName = '/record';
  const RecordScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            const SizedBox(height: 60),
            IconButton(
              icon: const Icon(Icons.arrow_back, color: Colors.black),
              onPressed: () => Navigator.pop(context),
            ),
            Text(
              'Graba tu mensaje',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            const SizedBox(height: 24),
            const Text(
              'Mantén presionado el botón para grabar',
              style: TextStyle(fontSize: 16, color: Colors.grey),
            ),
            const Spacer(),
            // Rectángulo GRIS de previsualización (como en Figma)
            Container(
              height: 250,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey[300],
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Center(
                child: Icon(Icons.videocam_off, size: 64, color: Colors.grey),
              ),
            ),
            const SizedBox(height: 48),
            // Botón ROJO CIRCULAR para grabar
            GestureDetector(
              onTapDown: (_) {
                // TODO: Iniciar grabación
                Navigator.pushNamed(context, '/translate');
              },
              onTapUp: (_) {
                // TODO: Parar grabación
              },
              child: Container(
                width: 80,
                height: 80,
                decoration: const BoxDecoration(
                  color: Colors.red,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: Colors.red,
                      blurRadius: 20,
                      spreadRadius: 2,
                    ),
                  ],
                ),
              ),
            ),
            const Spacer(),
          ],
        ),
      ),
    );
  }
}
