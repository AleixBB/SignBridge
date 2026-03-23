import 'package:flutter/material.dart';
import '../../core/widgets/primary_button.dart';

class TranslateScreen extends StatelessWidget {
  static const String routeName = '/translate';
  const TranslateScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            const SizedBox(height: 60),
            Text(
              'Mensaje traducido',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            const SizedBox(height: 24),
            // Vídeo/imagen del gesto (rectángulo del Figma)
            Container(
              height: 250,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey[300],
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Center(
                child: Icon(Icons.play_circle_outline, size: 64, color: Colors.grey),
              ),
            ),
            const SizedBox(height: 32),
            // Texto TRADUCIDO
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black12,
                    blurRadius: 10,
                    offset: const Offset(0, 5),
                  ),
                ],
              ),
              child: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Texto traducido:',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  SizedBox(height: 12),
                  Text(
                    '"Hola, ¿cómo estás?"',
                    style: TextStyle(fontSize: 18, height: 1.5),
                  ),
                ],
              ),
            ),
            const Spacer(),
            PrimaryButton(
              label: 'Guardar mensaje',
              onPressed: () {
                Navigator.pushNamed(context, '/saved');
              },
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}
