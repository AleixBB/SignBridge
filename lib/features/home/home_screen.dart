import 'package:flutter/material.dart';
import '../../core/widgets/primary_button.dart';

class HomeScreen extends StatelessWidget {
  static const String routeName = '/home';
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            const SizedBox(height: 60),
            // Saludo "Hola, User"
            Row(
              children: [
                Text(
                  'Hola, User',
                  style: Theme.of(context).textTheme.headlineMedium,
                ),
              ],
            ),
            const SizedBox(height: 8),
            const Text(
              '¿Qué quieres hacer hoy?',
              style: TextStyle(fontSize: 16, color: Colors.grey),
            ),
            const Spacer(),
            // Botón TRADUCIR (grande)
            PrimaryButton(
              label: 'Traducir',
              onPressed: () {
                // TODO: Ir a pantalla de grabación
                Navigator.pushNamed(context, '/record');
              },
            ),
            const SizedBox(height: 16),
            // Botón MENSAJES GUARDADOS
            PrimaryButton(
              label: 'Mensajes guardados',
              onPressed: () {
                // TODO: Ir a lista de mensajes
                Navigator.pushNamed(context, '/saved');
              },
            ),
            const Spacer(),
          ],
        ),
      ),
    );
  }
}
